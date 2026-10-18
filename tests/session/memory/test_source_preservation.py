# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import json

import pytest

from openviking.message import ImagePart, Message, TextPart
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
from openviking.session.memory.utils.source_metadata import (
    memory_origin_view,
    operation_source_metadata,
)
from openviking.storage.viking_fs import VikingFS
from openviking.utils.agfs_utils import RagfsBindingConfig, create_agfs_client
from openviking_cli.exceptions import NotFoundError
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.agfs_config import AGFSConfig
from openviking_cli.utils.config.open_viking_config import OpenVikingConfigSingleton
from tests.unit.test_message_origin import recorded_origin

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


def origin_message(message_id="msg-A", text="原文"):
    return Message.from_dict(
        {
            "id": message_id,
            "role": "user",
            "parts": [{"type": "text", "text": text}],
            "created_at": "2026-10-17T09:10:11.123456+07:00",
            "origin": recorded_origin(),
        }
    )


def test_origin_chunk_selection_uses_parent_and_actual_view():
    context = ExtractContext(
        [origin_message(text="A" * 110 + ". " + "B" * 110 + ". " + "C" * 110 + ".")]
    )
    selection = context.select_message_origins("1")
    assert selection.mode == "selected_ranges"
    assert selection.view_ranges == ((1, 1),)
    assert len(selection.messages) == 1
    selected = selection.messages[0]
    assert selected.facts.message_id == "msg-A"
    assert [(chunk.index, chunk.count) for chunk in selected.chunks] == [(1, 3)]
    assert (
        selected.facts.origin_statements.contributors[0].times[0].literal
        == "2026-10-17T09:10:11.123456+07:00"
    )
    assert context.messages[1].origin == context.messages[0].origin


async def test_image_projection_keeps_origin_without_a_model_call():
    from openviking.session.memory.vision_message_normalizer import (
        replace_image_parts_with_descriptions,
    )

    message = origin_message()
    message.parts.append(ImagePart(url="https://synthetic.invalid/image"))
    message.turn_id = "native-turn"
    projected = await replace_image_parts_with_descriptions([message], get_vlm=lambda: None)
    assert projected[0].origin == message.origin
    assert projected[0].created_at == message.created_at
    assert projected[0].turn_id == "native-turn"
    assert projected[0].parts == [TextPart("原文")]
    assert isinstance(message.parts[-1], ImagePart)


def test_origin_lawful_selections_share_run_without_false_conflict():
    context = ExtractContext([origin_message("first"), origin_message("second")])
    sources = [
        MemoryOperationSource(**A, message_origins=context.select_message_origins(str(i)))
        for i in (0, 1)
    ]
    op = operation()
    op.sources = sources
    fields = operation_source_metadata(op)
    assert len(fields["source_extractions"]) == 2
    assert "source_metadata_conflicts" not in fields
    assert "source_provenance_ambiguous" not in fields
    assert operation_source_metadata(op, old_fields=fields) == fields
    selected_ids = {
        source["message_origins"]["messages"][0]["facts"]["message_id"]
        for source in fields["source_extractions"]
    }
    assert selected_ids == {"first", "second"}


def test_origin_conflicting_known_literals_keep_variants_and_missing_witness():
    first = origin_message()
    altered = first.to_dict()
    altered["origin"]["contributors"][0]["times"][0]["literal"] = "2026-10-17T09:10:12.123456+07:00"
    op = operation()
    op.sources = [
        MemoryOperationSource.model_validate(A),
        MemoryOperationSource(
            **A, message_origins=ExtractContext([first]).select_message_origins()
        ),
        MemoryOperationSource(
            **A,
            message_origins=ExtractContext([Message.from_dict(altered)]).select_message_origins(),
        ),
    ]
    fields = operation_source_metadata(op)
    assert len(fields["source_extractions"]) == 3
    assert fields["source_metadata_conflicts"] == ["extract-a"]
    view = memory_origin_view(
        MemoryFile(uri=URI, memory_type="notes", content="derived", extra_fields=fields)
    )
    assert view.ambiguous is True
    assert any(conflict.field.endswith(".literal") for conflict in view.conflicts)
    assert any(not extraction.messages for extraction in view.extractions)
    assert view.exact_clause_proof is False
    assert view.semantic_withdrawal_established is False


async def test_origin_native_memory_persistence_and_reader_need_no_archive(native_target):
    fs, ctx, updater = native_target
    selection = ExtractContext([origin_message()]).select_message_origins()
    source = {**A, "message_origins": selection}
    for _ in range(2):
        result = await updater.apply_operations(
            ResolvedOperations(
                upsert_operations=[operation(source)], delete_file_contents=[], errors=[]
            ),
            ctx,
        )
        assert result.errors == []
    raw = await fs.read_file(URI, ctx=ctx)
    memory = MemoryFileUtils.read(raw, uri=URI)
    view = memory_origin_view(memory)
    assert len(view.extractions) == 1
    assert view.extractions[0].messages[0].facts.message_id == "msg-A"
    contributor = view.extractions[0].messages[0].facts.origin_statements.contributors[0]
    assert contributor.primary_author is None
    assert contributor.transport_sender is not None
    assert contributor.transport_sender.actor.identifier == "sender-A"
    assert contributor.times[0].literal == "2026-10-17T09:10:11.123456+07:00"
    assert "原文" not in raw
    source_selection = view.extractions[0].source.message_origins
    assert source_selection is not None
    assert source_selection.mode == "extraction_batch_candidates"


async def test_native_session_readback_archive_and_memory_keep_literal(native_target):
    from openviking.message.origin import _parse_origin_input
    from openviking.session.session import Session

    fs, ctx, updater = native_target
    session = Session(viking_fs=fs, ctx=ctx, session_id="native-origin")
    await session.ensure_exists()
    message = await session.add_message_async(
        "user",
        [TextPart("A" * 110 + ". " + "B" * 110 + ". " + "C" * 110 + ".")],
        created_at="2026-10-17T09:10:11.123456+07:00",
        source_message_ids=["record-A"],
        origin=_parse_origin_input(
            {key: value for key, value in recorded_origin().items() if key != "receiver"}
        ),
    )
    stored = json.loads(await fs.read_file(f"{session.uri}/messages.jsonl", ctx=ctx))
    assert stored == message.to_dict()
    archive_uri = f"{session.uri}/history/archive_literal"
    await fs.mkdir(archive_uri, exist_ok=True, ctx=ctx)
    await fs.write_file(f"{archive_uri}/messages.jsonl", message.to_jsonl() + "\n", ctx=ctx)
    restored = await session._archives.read_messages(archive_uri)
    selection = ExtractContext(restored).select_message_origins("1")
    result = await updater.apply_operations(
        ResolvedOperations(
            upsert_operations=[
                operation({**A, "archive_uri": archive_uri, "message_origins": selection})
            ],
            delete_file_contents=[],
            errors=[],
        ),
        ctx,
    )
    assert result.errors == []
    view = memory_origin_view(MemoryFileUtils.read(await fs.read_file(URI, ctx=ctx), uri=URI))
    selected = view.extractions[0].messages[0]
    assert selected.facts.message_id == message.id
    assert [(chunk.index, chunk.count) for chunk in selected.chunks] == [(1, 3)]
    assert selected.facts.origin == message.origin
    assert (
        selected.facts.origin_statements.contributors[0].times[0].literal
        == "2026-10-17T09:10:11.123456+07:00"
    )
    assert selected.facts.origin_statements.contributors[0].primary_author is None
    assert view.exact_clause_proof is False


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
    OpenVikingConfigSingleton.reset_instance()
    OpenVikingConfigSingleton.initialize(
        config_dict={
            "storage": {
                "workspace": str(tmp_path),
                "agfs": {"backend": "local"},
                "vectordb": {"backend": "local"},
            },
            "output_language_override": "en",
        }
    )
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
        OpenVikingConfigSingleton.reset_instance()


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
        MemoryField.model_validate(
            {"name": "source_extraction_id", "field_type": "string", "merge_op": "replace"}
        )
    )
    schema.fields.append(
        MemoryField.model_validate(
            {"name": "source_extractions", "field_type": "string", "merge_op": "replace"}
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
    forged = json.dumps(
        [
            {
                "extraction_id": "forged",
                "message_origins": {
                    "mode": "extraction_batch_candidates",
                    "messages": [{"facts": {"message_id": "forged-author"}}],
                },
            }
        ]
    )
    program = (
        f'sdk.create_notes(name="topic", content="Safe text.", source_extraction_id="forged", source_extractions={forged!r})'
        if protocol_name == "python"
        else json.dumps(
            {
                "notes": [
                    {
                        "page_id": 100,
                        "name": "topic",
                        "content": "Safe text.",
                        "source_extraction_id": "forged",
                        "source_extractions": forged,
                    }
                ]
            }
        )
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
    trusted = {**B, "message_origins": ExtractContext([origin_message()]).select_message_origins()}
    _inherit_source_metadata_to_merged_operations([operation(trusted)], resolved.upsert_operations)
    await updater._apply_upsert(resolved.upsert_operations[0], ctx)
    stored = MemoryFileUtils.read(await fs.read_file(URI, ctx=ctx))
    assert stored.extra_fields["source_extraction_id"] == "extract-b"
    view = memory_origin_view(stored)
    assert len(view.extractions) == 1
    assert view.extractions[0].source.extraction_id == "extract-b"
    assert view.extractions[0].messages[0].facts.message_id == "msg-A"
    assert (
        view.extractions[0].messages[0].facts.origin_statements.contributors[0].primary_author
        is None
    )
    assert "forged-author" not in json.dumps(stored.extra_fields)
