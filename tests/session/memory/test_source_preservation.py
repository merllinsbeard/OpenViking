# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Source preservation through native extraction, merge and local persistence."""

import json

import pytest

from openviking.message import Message, TextPart
from openviking.server.identity import RequestContext, Role, ToolContext
from openviking.session.memory.dataclass import (
    MemoryField,
    MemoryFile,
    MemoryOperationSource,
    MemoryTypeSchema,
    ResolvedOperation,
    ResolvedOperations,
)
from openviking.session.memory.extract_loop import ExtractLoop
from openviking.session.memory.extraction_output_protocol import (
    ExtractionOutputContext,
    create_extraction_output_protocol,
)
from openviking.session.memory.memory_isolation_handler import MemoryIsolationHandler
from openviking.session.memory.memory_type_registry import MemoryTypeRegistry
from openviking.session.memory.memory_updater import ExtractContext, MemoryUpdater
from openviking.session.memory.merge_op import FieldType, MergeOp
from openviking.session.memory.page_id_map import PageIdMap
from openviking.session.memory.patch_merge_context_provider import (
    PatchMergeContextProvider,
    PatchMergePatch,
    _render_field_diff_patches,
)
from openviking.session.memory.schema_model_generator import SchemaModelGenerator
from openviking.session.memory.streaming_memory_updater import (
    _inherit_source_metadata_to_merged_operations,
    render_operation_after_file,
)
from openviking.session.memory.tools import MemoryReadTool
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.storage.viking_fs import VikingFS
from openviking.utils.agfs_utils import RagfsBindingConfig, create_agfs_client
from openviking_cli.exceptions import NotFoundError
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.agfs_config import AGFSConfig

URI = "viking://user/default/memories/notes/topic.md"
RENAMED = "viking://user/default/memories/notes/renamed.md"
A = {
    "extraction_id": "extract-a",
    "session_id": "session-a",
    "archive_uri": "viking://session/session-a/archives/one",
    "task_id": "task-a",
    "trace_id": "trace-a",
    "extracted_at": "2026-01-02T03:04:05+07:00",
}
B = {
    "extraction_id": "extract-b",
    "session_id": "session-b",
    "archive_uri": "viking://session/session-b/archives/two",
    "task_id": "task-b",
    "trace_id": "trace-b",
    "extracted_at": "2026-01-03T04:05:06-04:00",
}


@pytest.fixture
def schema():
    return MemoryTypeSchema(
        memory_type="notes",
        directory="viking://user/{{ user_space }}/memories/notes",
        filename_template="{{ name }}.md",
        fields=[
            MemoryField(name="name", field_type=FieldType.STRING, merge_op=MergeOp.REPLACE),
            MemoryField(name="content", field_type=FieldType.STRING, merge_op=MergeOp.REPLACE),
        ],
    )


@pytest.fixture
def native_target(tmp_path, schema):
    client = create_agfs_client(
        RagfsBindingConfig(agfs=AGFSConfig(path=str(tmp_path / "storage"), backend="local"))
    )
    fs = VikingFS(agfs=client)
    ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.ROOT)
    registry = MemoryTypeRegistry(load_schemas=False)
    registry.register(schema)
    updater = MemoryUpdater(registry=registry)
    updater._viking_fs = fs
    try:
        yield fs, ctx, updater
    finally:
        client.close()


def operation(source=None, *, uri=URI, old=None, **fields):
    return ResolvedOperation(
        memory_type="notes",
        memory_fields={"name": "topic", "content": "Updated text.", **fields},
        uris=[uri],
        source=MemoryOperationSource(**source) if source is not None else None,
        old_memory_file_content=old,
    )


def test_chunk_round_trip_preserves_supplied_message_fields():
    message = Message(
        id="msg-original",
        role="assistant",
        parts=[TextPart("A" * 110 + ". " + "B" * 110 + ".")],
        peer_id="peer-a",
        created_at="2026-01-01T03:04:05.678+07:00",
        turn_id="turn-a",
        message_kind="tool_transport",
        source_message_ids=["provider-1", "provider-2"],
    )
    context = ExtractContext([Message.from_dict(json.loads(message.to_jsonl()))])
    assert [m.content for m in context.messages] == ["A" * 110 + ".", " " + "B" * 110 + "."]
    for i, chunk in enumerate(context.messages):
        assert Message.from_dict(json.loads(chunk.to_jsonl())).to_dict() == {
            "id": f"msg-original#chunk_{i}",
            "role": "assistant",
            "parts": [{"type": "text", "text": chunk.content}],
            "peer_id": "peer-a",
            "created_at": "2026-01-01T03:04:05.678+07:00",
            "turn_id": "turn-a",
            "message_kind": "tool_transport",
            "source_message_ids": ["provider-1", "provider-2"],
        }
        assert context.chunk_meta[id(chunk)].source_message_id == "msg-original"
    assert message.source_message_ids == ["provider-1", "provider-2"]


def test_chunk_does_not_invent_missing_message_metadata():
    context = ExtractContext(
        [
            Message(
                id="msg-unknown", role="user", parts=[TextPart("A" * 110 + ". " + "B" * 110 + ".")]
            )
        ]
    )
    assert len(context.messages) == 2
    assert [
        (m.turn_id, m.message_kind, m.source_message_ids, m.created_at) for m in context.messages
    ] == [
        (None, None, None, None),
        (None, None, None, None),
    ]


@pytest.mark.parametrize("sources", [[A], [B, A, A]])
async def test_exact_merge_serializes_full_sources(schema, sources):
    merged = operation()
    _inherit_source_metadata_to_merged_operations(
        [*(operation(s) for s in sources), operation(B, uri=RENAMED)], [merged]
    )
    after = await render_operation_after_file(
        merged, schema=schema, extract_context=ExtractContext([])
    )
    fields = MemoryFileUtils.read(MemoryFileUtils.write(after)).extra_fields
    expected = [A] if sources == [A] else [A, B]
    assert fields["source_extractions"] == expected
    assert fields.get("source_provenance_ambiguous", False) is False
    assert fields.get("source_metadata_conflicts", []) == []
    if len(expected) == 1:
        assert fields["source_extraction_id"] == "extract-a"
    else:
        assert fields["source_extraction_ids"] == ["extract-a", "extract-b"]


@pytest.mark.parametrize("sources", [[A], [A, B]])
async def test_unmatched_merge_keeps_candidate_scope(schema, sources):
    merged = operation(uri=RENAMED)
    _inherit_source_metadata_to_merged_operations([operation(s) for s in sources], [merged])
    after = await render_operation_after_file(
        merged, schema=schema, extract_context=ExtractContext([])
    )
    assert after.extra_fields["source_extractions"] == sources
    assert after.extra_fields["source_provenance_ambiguous"] is True


async def test_legacy_id_never_fabricates_source_details(schema):
    merged = operation(uri=RENAMED)
    _inherit_source_metadata_to_merged_operations(
        [operation(source_extraction_id="legacy-only")], [merged]
    )
    after = await render_operation_after_file(
        merged, schema=schema, extract_context=ExtractContext([])
    )
    assert after.extra_fields["source_extraction_id"] == "legacy-only"
    assert after.extra_fields.get("source_extractions", []) == []
    assert after.extra_fields["source_provenance_ambiguous"] is True


async def test_mixed_legacy_and_full_sources_survive_a_second_merge(schema):
    legacy = operation(source_extraction_id="legacy-only")
    first = operation(uri=RENAMED)
    _inherit_source_metadata_to_merged_operations([legacy, operation(A)], [first])
    second = operation(uri=RENAMED)
    _inherit_source_metadata_to_merged_operations([first], [second])
    after = await render_operation_after_file(
        second, schema=schema, extract_context=ExtractContext([])
    )
    assert after.extra_fields["source_extractions"] == [A]
    assert after.extra_fields["source_extraction_ids"] == ["extract-a", "legacy-only"]
    assert after.extra_fields["source_provenance_ambiguous"] is True


async def test_partly_unmatched_output_remains_ambiguous(schema):
    merged = operation()
    merged.uris.append(RENAMED)
    _inherit_source_metadata_to_merged_operations([operation(A)], [merged])
    after = await render_operation_after_file(
        merged, schema=schema, extract_context=ExtractContext([])
    )
    assert after.extra_fields["source_extractions"] == [A]
    assert after.extra_fields["source_provenance_ambiguous"] is True


async def test_conflicting_same_id_retains_both_snapshots(schema):
    conflicting = {**A, "session_id": "session-conflict"}
    outputs = []
    for inputs in ([A, conflicting, A], [conflicting, A, conflicting]):
        merged = operation()
        _inherit_source_metadata_to_merged_operations([operation(s) for s in inputs], [merged])
        after = await render_operation_after_file(
            merged, schema=schema, extract_context=ExtractContext([])
        )
        outputs.append(after.extra_fields)
    assert outputs[0] == outputs[1]
    assert outputs[0]["source_extractions"] == [A, conflicting]
    assert outputs[0]["source_metadata_conflicts"] == ["extract-a"]
    assert outputs[0]["source_provenance_ambiguous"] is True


async def test_native_persistence_unions_replay_and_rename(native_target):
    fs, ctx, updater = native_target
    old = MemoryFile(
        uri=URI,
        memory_type="notes",
        content="Original text.",
        extra_fields={
            "name": "topic",
            "source_extractions": [A],
            "source_extraction_id": "extract-a",
            "old_only": "keep this value",
            "version": 3,
        },
    )
    await fs.write_file(URI, MemoryFileUtils.write(old), ctx=ctx)
    for version in (4, 5):
        result = await updater.apply_operations(
            ResolvedOperations(
                upsert_operations=[operation(B)], delete_file_contents=[], errors=[]
            ),
            ctx,
        )
        assert result.errors == []
        raw = await fs.read_file(URI, ctx=ctx)
        fields = json.loads(raw.split("<!-- MEMORY_FIELDS\n", 1)[1].removesuffix("\n-->"))
        assert fields["source_extractions"] == [A, B]
        assert fields["source_extraction_ids"] == ["extract-a", "extract-b"]
        assert fields["old_only"] == "keep this value"
        assert fields["last_update_trace_id"] == "trace-b"
        assert fields["version"] == version
        assert raw.startswith("Updated text.\n\n<!-- MEMORY_FIELDS\n")
    snapshot = MemoryFileUtils.read(raw, uri=URI)
    result = await updater.apply_operations(
        ResolvedOperations(
            upsert_operations=[operation(uri=RENAMED, old=snapshot, name="renamed")],
            delete_file_contents=[],
            errors=[],
        ),
        ctx,
    )
    assert result.errors == []
    assert result.deleted_uris == [URI]
    with pytest.raises(NotFoundError):
        await fs.read_file(URI, ctx=ctx)
    renamed = MemoryFileUtils.read(await fs.read_file(RENAMED, ctx=ctx), uri=RENAMED)
    assert renamed.extra_fields["source_extractions"] == [A, B]
    assert renamed.extra_fields["source_extraction_ids"] == ["extract-a", "extract-b"]
    assert renamed.extra_fields["old_only"] == "keep this value"
    assert renamed.extra_fields["name"] == "renamed"
    assert renamed.content == "Updated text."


async def test_system_sources_are_hidden_and_cannot_be_forged(native_target, schema):
    fs, ctx, updater = native_target
    old = MemoryFile(
        uri=URI,
        memory_type="notes",
        content="Original text.",
        extra_fields={
            "name": "topic",
            "source_extractions": [A],
            "source_provenance_ambiguous": True,
            "source_metadata_conflicts": ["extract-a"],
            "source_extraction_id": "extract-a",
        },
    )
    await fs.write_file(URI, MemoryFileUtils.write(old), ctx=ctx)
    page_ids = PageIdMap()
    tool_ctx = ToolContext(fs, ctx, read_file_contents={}, page_id_map=page_ids)
    visible = await MemoryReadTool().execute(tool_ctx, uri=URI)
    assert visible["content"] == "1\tOriginal text."
    assert "source_extractions" not in visible
    assert "source_provenance_ambiguous" not in visible
    assert "source_metadata_conflicts" not in visible
    diff = _render_field_diff_patches([PatchMergePatch(None, old)])
    assert "Original text." in diff
    assert "source_extractions" not in diff
    assert "source_provenance_ambiguous" not in diff
    protocol = create_extraction_output_protocol("python")
    context = ExtractionOutputContext(
        operations_model=SchemaModelGenerator([schema]).create_structured_operations_model(),
        schemas=(schema,),
        page_id_map=page_ids,
        read_file_contents=tool_ctx.read_file_contents,
        link_enabled=False,
    )
    bindings = protocol.render_new_bindings(context, source="read")
    assert "content='Original text.'" in bindings
    assert "source_extractions" not in bindings
    name = bindings.split(" = sdk.existing(", 1)[0].strip().splitlines()[-1]
    parsed, error = protocol.parse(
        f'{name}.update(content="Safe text.", source_extractions=[{{"extraction_id": "forged"}}], '
        'source_extraction_id="forged", source_provenance_ambiguous=False)\n'
        f'{name}.source_extractions.update("forged")\nsdk.commit()',
        context,
    )
    assert error is None
    assert parsed.notes[0].model_dump(exclude_none=True) == {
        "page_id": 1,
        "name": "topic",
        "content": "Safe text.",
    }
    provider = PatchMergeContextProvider(
        memory_type="notes", patches=[], memory_registry=updater._registry
    )
    provider.read_file_contents.update(tool_ctx.read_file_contents)
    extract_context = provider.get_extract_context()
    extract_context.page_id_map = page_ids
    loop = ExtractLoop(
        vlm=None,
        model="unused",
        viking_fs=fs,
        ctx=ctx,
        context_provider=provider,
        isolation_handler=MemoryIsolationHandler(ctx, extract_context),
    )
    loop._extract_context = extract_context
    resolved, links = await loop.resolve_operations(parsed)
    assert resolved.errors == []
    assert links == []
    assert resolved.upsert_operations[0].uris == [URI]
    _inherit_source_metadata_to_merged_operations([operation(B)], resolved.upsert_operations)
    await updater._apply_upsert(resolved.upsert_operations[0], ctx)
    stored = MemoryFileUtils.read(await fs.read_file(URI, ctx=ctx))
    assert stored.extra_fields["source_extractions"] == [A, B]
    assert stored.extra_fields["source_provenance_ambiguous"] is True


@pytest.mark.parametrize("protocol_name", ["python", "json"])
async def test_model_cannot_supply_system_fields_even_when_schema_lists_them(
    native_target, schema, protocol_name
):
    fs, ctx, updater = native_target
    schema.fields.append(
        MemoryField(
            name="source_extraction_id", field_type=FieldType.STRING, merge_op=MergeOp.REPLACE
        )
    )
    provider = PatchMergeContextProvider(
        memory_type="notes", patches=[], memory_registry=updater._registry
    )
    extract_context = provider.get_extract_context()
    protocol = create_extraction_output_protocol(protocol_name)
    context = ExtractionOutputContext(
        operations_model=SchemaModelGenerator([schema]).create_structured_operations_model(),
        schemas=(schema,),
        page_id_map=extract_context.page_id_map,
        read_file_contents={},
        link_enabled=False,
    )
    program = (
        'sdk.create_notes(name="topic", content="Safe text.", source_extraction_id="forged")'
        if protocol_name == "python"
        else '{"notes": [{"page_id": 100, "name": "topic", "content": "Safe text.", '
        '"source_extraction_id": "forged"}]}'
    )
    parsed, error = protocol.parse(program, context)
    assert error is None
    loop = ExtractLoop(
        vlm=None,
        model="unused",
        viking_fs=fs,
        ctx=ctx,
        context_provider=provider,
        isolation_handler=MemoryIsolationHandler(ctx, extract_context),
    )
    loop._extract_context = extract_context
    resolved, _ = await loop.resolve_operations(parsed)
    assert resolved.errors == []
    assert resolved.upsert_operations[0].memory_fields == {
        "name": "topic",
        "content": "Safe text.",
        "memory_type": "notes",
        "user_id": "default",
    }
    _inherit_source_metadata_to_merged_operations([operation(B)], resolved.upsert_operations)
    await updater._apply_upsert(resolved.upsert_operations[0], ctx)
    stored = MemoryFileUtils.read(await fs.read_file(URI, ctx=ctx))
    assert stored.extra_fields["source_extraction_id"] == "extract-b"
    assert stored.extra_fields["source_extractions"] == [B]
