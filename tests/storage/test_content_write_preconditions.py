# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import hashlib

import pytest

from openviking.config.binding import manager_over_source
from openviking.config.source.memory_source import MemoryConfigSource
from openviking.config.vector import AccountVectorConfigResolver
from openviking.server.identity import RequestContext, Role
from openviking.storage import viking_fs as fs_module
from openviking.storage.content_write import ContentWriteCoordinator
from openviking.storage.errors import ResourceBusyError
from openviking.storage.queuefs import init_queue_manager
from openviking.storage.queuefs import queue_manager as queue_module
from openviking.storage.viking_fs import init_viking_fs
from openviking.utils.agfs_utils import RagfsBindingConfig, create_agfs_client
from openviking_cli.exceptions import (
    AlreadyExistsError,
    ConflictError,
    InvalidArgumentError,
    NotFoundError,
    OpenVikingError,
)
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.agfs_config import AGFSConfig
from openviking_cli.utils.config.open_viking_config import (
    OpenVikingConfig,
    OpenVikingConfigSingleton,
)

ROOT = "viking://resources/contract"


def digest(content):
    return hashlib.sha256(content.encode() if isinstance(content, str) else content).hexdigest()


@pytest.fixture
async def native_target(tmp_path, monkeypatch):
    config_path = tmp_path / "ov.conf"
    config_path.write_text("{}")
    monkeypatch.setenv("OPENVIKING_CONFIG_FILE", str(config_path))
    previous_fs = fs_module._instance
    previous_queue = queue_module._instance
    previous_config = OpenVikingConfigSingleton._instance
    config = OpenVikingConfig.from_dict({"storage": {"workspace": str(tmp_path)}})
    OpenVikingConfigSingleton._instance = config
    client = create_agfs_client(
        RagfsBindingConfig(agfs=AGFSConfig(path=str(tmp_path / "storage"), backend="local"))
    )
    manager = manager_over_source(MemoryConfigSource(), base_config=config)
    await manager.initialize()
    fs = init_viking_fs(agfs=client, vector_config_resolver=AccountVectorConfigResolver(manager))
    queue = init_queue_manager(client)
    queue.get_queue(queue.EMBEDDING, allow_create=True)
    queue.get_queue(queue.SEMANTIC, allow_create=True)
    ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.ROOT)
    await fs.mkdir(ROOT, ctx=ctx, exist_ok=True)
    coordinator = ContentWriteCoordinator(fs)
    try:
        yield fs, ctx, coordinator
    finally:
        fs_module._instance = previous_fs
        queue_module._instance = previous_queue
        OpenVikingConfigSingleton._instance = previous_config
        client.close()


async def write(target, operations, **options):
    _, ctx, coordinator = target
    return await coordinator.batch_write(
        root_uri=ROOT,
        operations=operations,
        ctx=ctx,
        wait=False,
        processing_mode="vectors_only",
        **options,
    )


async def test_create_preserves_occupied_target_and_materializes_missing_target(native_target):
    fs, ctx, _ = native_target
    occupied = ROOT + "/occupied.md"
    new = ROOT + "/new.md"
    await fs.write_file(occupied, "other writer", ctx=ctx)
    result = await write(
        native_target,
        [
            {"uri": occupied, "content": "stale proposal", "mode": "create"},
            {"uri": new, "content": "new bytes", "mode": "create"},
        ],
        skip_conflicts=True,
    )
    assert result["created"] == [new]
    assert result["conflicts"][0]["uri"] == occupied
    assert result["conflicts"][0]["code"] == "ALREADY_EXISTS"
    assert await fs.read_file(occupied, ctx=ctx) == "other writer"
    assert await fs.read_file(new, ctx=ctx) == "new bytes"
    with pytest.raises(AlreadyExistsError):
        await write(native_target, [{"uri": occupied, "content": "overwrite", "mode": "create"}])
    assert await fs.read_file(occupied, ctx=ctx) == "other writer"


async def test_stale_revision_rejects_entire_preflight_before_any_write(native_target):
    fs, ctx, _ = native_target
    first, stale = ROOT + "/a.md", ROOT + "/z.md"
    await fs.write_file(first, "first", ctx=ctx)
    await fs.write_file(stale, "newer", ctx=ctx)
    with pytest.raises(ConflictError):
        await write(
            native_target,
            [
                {"uri": first, "content": "changed", "expected_sha256": digest("first")},
                {"uri": stale, "content": "stale", "expected_sha256": digest("old")},
            ],
        )
    assert await fs.read_file(first, ctx=ctx) == "first"
    assert await fs.read_file(stale, ctx=ctx) == "newer"


@pytest.mark.parametrize("skip_conflicts", [False, True])
@pytest.mark.parametrize("filename", ["missing.md", ".abstract.md", ".overview.md"])
@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("replace", None),
        ("replace", digest("old")),
        ("append", None),
        ("append", digest("old")),
        ("upsert", digest("old")),
    ],
)
async def test_missing_target_conflicts_preserve_absence_and_allow_compatible_writes(
    native_target, skip_conflicts, filename, mode, expected
):
    fs, ctx, _ = native_target
    uri = ROOT + "/" + filename
    compatible = ROOT + "/compatible.md"
    operations = [
        {"uri": compatible, "content": "compatible bytes", "mode": "create"},
        {"uri": uri, "content": "resurrected", "mode": mode, "expected_sha256": expected},
    ]
    error_type = ConflictError if mode == "upsert" else NotFoundError
    if skip_conflicts:
        result = await write(native_target, operations, skip_conflicts=True)
        assert result["created"] == [compatible]
        assert result["updated"] == []
        assert result["conflicts"][0]["uri"] == uri
        assert result["conflicts"][0]["code"] == ("CONFLICT" if mode == "upsert" else "NOT_FOUND")
        assert len(result["conflicts"]) == 1
        assert await fs.read_file(compatible, ctx=ctx) == "compatible bytes"
    else:
        with pytest.raises(error_type):
            await write(native_target, operations, skip_conflicts=False)
        with pytest.raises(NotFoundError):
            await fs.read_file(compatible, ctx=ctx)
    with pytest.raises(NotFoundError):
        await fs.read_file(uri, ctx=ctx)
    queue = queue_module.get_queue_manager()
    assert await queue.size(queue.EMBEDDING) == int(skip_conflicts)
    assert await queue.size(queue.SEMANTIC) == 0
    if skip_conflicts:
        message = await queue.get_queue(queue.EMBEDDING).peek()
        assert message.context_data["uri"] == compatible
        assert message.payload.message == "compatible bytes"


@pytest.mark.parametrize("skip_conflicts", [False, True])
@pytest.mark.parametrize("filename", [".abstract.md", ".overview.md"])
@pytest.mark.parametrize("mode", ["create", "upsert"])
async def test_generated_sidecar_creation_is_forbidden_even_with_skip_conflicts(
    native_target, skip_conflicts, filename, mode
):
    fs, ctx, _ = native_target
    uri = ROOT + "/" + filename
    compatible = ROOT + "/compatible.md"
    with pytest.raises(InvalidArgumentError, match="cannot create generated"):
        await write(
            native_target,
            [
                {"uri": compatible, "content": "compatible bytes", "mode": "create"},
                {"uri": uri, "content": "forbidden", "mode": mode},
            ],
            skip_conflicts=skip_conflicts,
        )
    for target_uri in (uri, compatible):
        with pytest.raises(NotFoundError):
            await fs.read_file(target_uri, ctx=ctx)
    queue = queue_module.get_queue_manager()
    assert await queue.size(queue.EMBEDDING) == 0
    assert await queue.size(queue.SEMANTIC) == 0
    result = await write(
        native_target, [{"uri": compatible, "content": "valid bytes", "mode": "create"}]
    )
    assert result["created"] == [compatible]
    assert await fs.read_file(compatible, ctx=ctx) == "valid bytes"


async def test_parallel_writers_cannot_both_replace_the_same_revision(native_target):
    fs, ctx, _ = native_target
    uri = ROOT + "/parallel.md"
    await fs.write_file(uri, "base", ctx=ctx)
    gate = asyncio.Event()

    async def propose(content):
        await gate.wait()
        return await write(
            native_target, [{"uri": uri, "content": content, "expected_sha256": digest("base")}]
        )

    tasks = [asyncio.create_task(propose(content)) for content in ("writer A", "writer B")]
    gate.set()
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    assert sum(isinstance(value, dict) for value in outcomes) == 1
    failures = [value for value in outcomes if isinstance(value, Exception)]
    assert len(failures) == 1
    assert isinstance(failures[0], (ConflictError, ResourceBusyError))
    winner = await fs.read_file(uri, ctx=ctx)
    assert winner in {"writer A", "writer B"}
    with pytest.raises(ConflictError):
        await write(
            native_target,
            [{"uri": uri, "content": "late retry", "expected_sha256": digest("base")}],
        )
    assert await fs.read_file(uri, ctx=ctx) == winner


async def test_parallel_creators_preserve_one_winner(native_target):
    fs, ctx, _ = native_target
    uri = ROOT + "/new-parallel.md"
    outcomes = await asyncio.gather(
        *(
            write(native_target, [{"uri": uri, "content": content, "mode": "create"}])
            for content in ("A", "B")
        ),
        return_exceptions=True,
    )
    assert sum(isinstance(value, dict) for value in outcomes) == 1
    assert any(isinstance(value, (AlreadyExistsError, ResourceBusyError)) for value in outcomes)
    assert await fs.read_file(uri, ctx=ctx) in {"A", "B"}


async def test_lost_ack_requires_exact_reconciliation_before_replay(native_target):
    fs, ctx, _ = native_target
    uri = ROOT + "/ack.md"
    desired = "new bytes\n"
    await fs.write_file(uri, "old", ctx=ctx)
    operation = {"uri": uri, "content": desired, "expected_sha256": digest("old")}
    await write(native_target, [operation])
    with pytest.raises(ConflictError):
        await write(native_target, [operation])
    assert await fs.read_file_bytes(uri, ctx=ctx) == b"new bytes\n"
    result = await write(
        native_target,
        [{**operation, "expected_sha256": digest(desired)}],
        skip_conflicts=True,
    )
    assert result["unchanged"] == [uri]
    assert result["conflicts"] == []
    await fs.write_file(uri, "third-party bytes", ctx=ctx)
    with pytest.raises(ConflictError):
        await write(native_target, [operation])
    assert await fs.read_file_bytes(uri, ctx=ctx) == b"third-party bytes"


async def test_append_replay_with_old_hash_cannot_duplicate_content(native_target):
    fs, ctx, _ = native_target
    uri = ROOT + "/append.md"
    await fs.write_file(uri, "prefix\n", ctx=ctx)
    operation = {
        "uri": uri,
        "content": "suffix\n",
        "mode": "append",
        "expected_sha256": digest("prefix\n"),
    }
    await write(native_target, [operation])
    with pytest.raises(ConflictError):
        await write(native_target, [operation])
    assert await fs.read_file_bytes(uri, ctx=ctx) == b"prefix\nsuffix\n"
    await write(native_target, [{"uri": uri, "content": "suffix\n", "mode": "append"}])
    assert await fs.read_file_bytes(uri, ctx=ctx) == b"prefix\nsuffix\nsuffix\n"


async def test_binary_revision_and_literal_uri_with_spaces(native_target):
    fs, ctx, _ = native_target
    uri = ROOT + "/exact name.bin"
    await fs.write_file(uri, b"\x00\xffold", ctx=ctx)
    result = await write(
        native_target,
        [
            {
                "uri": uri,
                "content_base64": base64.b64encode(b"\xff\x00new").decode(),
                "expected_sha256": digest(b"\x00\xffold"),
            }
        ],
    )
    assert result["updated"] == [uri]
    assert await fs.read_file_bytes(uri, ctx=ctx) == b"\xff\x00new"
    with pytest.raises(NotFoundError):
        await fs.read_file(uri.replace("exact name", "exact-name"), ctx=ctx)


async def test_io_failure_leaves_earlier_effect_visible_and_later_file_untouched(
    native_target, monkeypatch
):
    fs, ctx, coordinator = native_target
    first, second = ROOT + "/a.md", ROOT + "/b.md"
    for uri in (first, second):
        await fs.write_file(uri, "before", ctx=ctx)
    original = coordinator._write_in_place

    async def fail_second(uri, content, **kwargs):
        if uri == second:
            raise OSError("injected second-file failure")
        return await original(uri, content, **kwargs)

    monkeypatch.setattr(coordinator, "_write_in_place", fail_second)
    with pytest.raises(OSError, match="second-file"):
        await write(
            native_target,
            [
                {"uri": uri, "content": "after", "expected_sha256": digest("before")}
                for uri in (first, second)
            ],
        )
    assert await fs.read_file(first, ctx=ctx) == "after"
    assert await fs.read_file(second, ctx=ctx) == "before"
    monkeypatch.setattr(coordinator, "_write_in_place", original)
    result = await write(
        native_target, [{"uri": second, "content": "after", "expected_sha256": digest("before")}]
    )
    assert result["updated"] == [second]
    assert await fs.read_file(second, ctx=ctx) == "after"


async def test_refresh_failure_keeps_content_and_replay_can_retry_refresh(
    native_target, monkeypatch
):
    fs, ctx, coordinator = native_target
    uri = ROOT + "/refresh.md"
    await fs.write_file(uri, "before", ctx=ctx)
    original_refresh = coordinator._refresh_batch

    async def fail_refresh(**kwargs):
        raise RuntimeError("injected refresh failure")

    monkeypatch.setattr(coordinator, "_refresh_batch", fail_refresh)
    with pytest.raises(OpenVikingError) as error:
        await write(
            native_target, [{"uri": uri, "content": "after", "expected_sha256": digest("before")}]
        )
    assert error.value.code == "REFRESH_FAILED"
    assert error.value.details["updated"] == [uri]
    assert await fs.read_file(uri, ctx=ctx) == "after"
    skipped = await write(
        native_target,
        [{"uri": uri, "content": "after", "expected_sha256": digest("after")}],
        skip_conflicts=True,
    )
    assert skipped["unchanged"] == [uri]
    queue = queue_module.get_queue_manager()
    assert await queue.size(queue.SEMANTIC) == 0
    monkeypatch.setattr(coordinator, "_refresh_batch", original_refresh)
    result = await coordinator.batch_write(
        root_uri=ROOT,
        ctx=ctx,
        operations=[{"uri": uri, "content": "after", "expected_sha256": digest("after")}],
        wait=False,
        processing_mode="semantic_and_vectors",
        skip_conflicts=False,
    )
    assert result["updated"] == [uri]
    assert result["semantic_status"] == "queued"
    semantic = await queue.get_queue(queue.SEMANTIC).peek()
    assert semantic.changes == {"modified": [uri]}
    assert await fs.read_file(uri, ctx=ctx) == "after"


async def test_io_error_after_a_partial_file_write_requires_reconciliation(
    native_target, monkeypatch
):
    fs, ctx, coordinator = native_target
    uri = ROOT + "/partial.md"
    await fs.write_file(uri, "before", ctx=ctx)
    original = coordinator._write_in_place

    async def partial_write(target_uri, content, **kwargs):
        await original(target_uri, "part", **kwargs)
        raise OSError("injected error after partial bytes")

    monkeypatch.setattr(coordinator, "_write_in_place", partial_write)
    operation = {"uri": uri, "content": "after", "expected_sha256": digest("before")}
    with pytest.raises(OSError, match="partial bytes"):
        await write(native_target, [operation])
    assert await fs.read_file_bytes(uri, ctx=ctx) == b"part"
    monkeypatch.setattr(coordinator, "_write_in_place", original)
    with pytest.raises(ConflictError):
        await write(native_target, [operation])
    assert await fs.read_file_bytes(uri, ctx=ctx) == b"part"


async def test_preconditions_are_optional_and_raw_writers_bypass_them(native_target):
    fs, ctx, _ = native_target
    uri = ROOT + "/optional.md"
    await fs.write_file(uri, "newer", ctx=ctx)
    await write(native_target, [{"uri": uri, "content": "unqualified overwrite"}])
    assert await fs.read_file(uri, ctx=ctx) == "unqualified overwrite"
    await fs.write_file(uri, "raw overwrite", ctx=ctx)
    assert await fs.read_file(uri, ctx=ctx) == "raw overwrite"


async def test_vectors_only_queues_exact_new_content_and_leaves_derived_bytes_stale(native_target):
    fs, ctx, coordinator = native_target
    uri = ROOT + "/exact name.md"
    await fs.write_file(uri, "old source", ctx=ctx)
    await fs.write_file(ROOT + "/.abstract.md", "old abstract", ctx=ctx)
    await fs.write_file(ROOT + "/.overview.md", "old overview", ctx=ctx)
    result = await write(
        native_target,
        [{"uri": uri, "content": "new source", "expected_sha256": digest("old source")}],
    )
    assert result["semantic_status"] == "skipped"
    assert result["vector_status"] == "queued"
    assert await fs.read_file(uri, ctx=ctx) == "new source"
    assert await fs.read_file(ROOT + "/.abstract.md", ctx=ctx) == "old abstract"
    assert await fs.read_file(ROOT + "/.overview.md", ctx=ctx) == "old overview"
    queue = queue_module.get_queue_manager()
    message = await queue.get_queue(queue.EMBEDDING).peek()
    assert message.context_data["uri"] == uri
    assert message.payload.message == "new source"
    assert await queue.size(queue.SEMANTIC) == 0
    refreshed = await coordinator.batch_write(
        root_uri=ROOT,
        ctx=ctx,
        operations=[{"uri": uri, "content": "new source", "expected_sha256": digest("new source")}],
        wait=False,
    )
    assert refreshed["semantic_status"] == "queued"
    semantic = await queue.get_queue(queue.SEMANTIC).peek()
    assert semantic.uri == ROOT
    assert semantic.changes == {"modified": [uri]}
    assert await fs.read_file(ROOT + "/.overview.md", ctx=ctx) == "old overview"


async def test_vector_payload_keeps_its_revision_when_a_later_writer_commits_before_refresh(
    native_target, monkeypatch
):
    fs, ctx, coordinator = native_target
    uri = ROOT + "/interleaved.md"
    await fs.write_file(uri, "base", ctx=ctx)
    original_refresh = coordinator._refresh_batch

    async def interleave(**kwargs):
        other = ContentWriteCoordinator(fs)
        await other.batch_write(
            root_uri=ROOT,
            operations=[{"uri": uri, "content": "writer B", "expected_sha256": digest("writer A")}],
            ctx=ctx,
            wait=False,
            processing_mode="vectors_only",
        )
        return await original_refresh(**kwargs)

    monkeypatch.setattr(coordinator, "_refresh_batch", interleave)
    await write(
        native_target, [{"uri": uri, "content": "writer A", "expected_sha256": digest("base")}]
    )
    queue = queue_module.get_queue_manager().get_queue("Embedding")
    messages = [await queue.dequeue(), await queue.dequeue()]
    assert [message.payload.message for message in messages] == ["writer B", "writer A"]
    assert [message.context_data["_materialized_content"] for message in messages] == [
        "writer B",
        "writer A",
    ]
    assert [message.context_data["md5"] for message in messages] == [
        hashlib.md5(b"writer B").hexdigest(),
        hashlib.md5(b"writer A").hexdigest(),
    ]
    assert await fs.read_file(uri, ctx=ctx) == "writer B"


async def test_http_contract_reads_exact_uri_and_rejects_stale_revision(native_target):
    from types import SimpleNamespace

    import httpx

    from openviking.server.app import create_app
    from openviking.server.auth import get_request_context
    from openviking.server.config import ServerConfig
    from openviking.server.dependencies import get_service_or_none, set_service
    from openviking.service.fs_service import FSService

    fs, ctx, _ = native_target
    uri = ROOT + "/exact name.md"
    await fs.write_file(uri, "old", ctx=ctx)
    previous_service = get_service_or_none()
    set_service(SimpleNamespace(fs=FSService(viking_fs=fs)))
    app = create_app(config=ServerConfig())
    app.dependency_overrides[get_request_context] = lambda: ctx
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            payload = {
                "root_uri": ROOT,
                "operations": [{"uri": uri, "content": "new", "expected_sha256": digest("old")}],
                "wait": False,
                "processing_mode": "vectors_only",
            }
            response = await client.post("/api/v1/content/batch-write", json=payload)
            assert response.status_code == 200, response.text
            assert response.json()["result"]["updated"] == [uri]
            replay = await client.post("/api/v1/content/batch-write", json=payload)
            assert replay.status_code == 409
            assert replay.json()["error"]["code"] == "CONFLICT"
            readback = await client.get(
                "/api/v1/content/read", params={"uri": uri, "raw": True, "offset": 0, "limit": -1}
            )
            assert readback.status_code == 200
            assert readback.json()["result"] == "new"
            malformed = await client.post(
                "/api/v1/content/batch-write",
                json={
                    **payload,
                    "operations": [{"uri": uri, "content": "bad", "expected_sha256": "bad"}],
                },
            )
            assert malformed.status_code == 400
            assert await fs.read_file(uri, ctx=ctx) == "new"
            from openviking_sdk import AsyncHTTPClient

            sdk = AsyncHTTPClient(url="http://test", api_key="test-only")
            sdk._http = client
            sdk_result = await sdk.batch_write(
                ROOT,
                [{"uri": uri, "content": "SDK bytes", "expected_sha256": digest("new")}],
                wait=False,
                options={"extra": {"processing_mode": "vectors_only", "skip_conflicts": True}},
            )
            assert sdk_result["updated"] == [uri]
            assert sdk_result["conflicts"] == []
            assert await fs.read_file(uri, ctx=ctx) == "SDK bytes"
    finally:
        set_service(previous_service)


@pytest.mark.parametrize("value", ["", "A" * 64, "g" * 64, "0" * 63, 123])
async def test_invalid_revision_cannot_write(native_target, value):
    fs, ctx, _ = native_target
    uri = ROOT + "/revision.md"
    await fs.write_file(uri, "before", ctx=ctx)
    with pytest.raises(InvalidArgumentError):
        await write(native_target, [{"uri": uri, "content": "after", "expected_sha256": value}])
    assert await fs.read_file(uri, ctx=ctx) == "before"
