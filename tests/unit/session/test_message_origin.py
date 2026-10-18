# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Native Session and archive behavior with explicitly MOCK storage transport."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI

from openviking.message import TextPart, ToolPart
from openviking.message.origin import OriginContractMismatch, _parse_origin_input
from openviking.session.session import Session
from openviking.storage.viking_fs import VikingFS
from tests.unit.session.test_session_commit_resume import _MemoryVikingFS
from tests.unit.test_message_origin import source_statements


@pytest.fixture
def origin_session():
    uri = "viking://user/default/sessions/origin-test"
    storage = _MemoryVikingFS({})
    transport = Mock(spec=VikingFS, wraps=storage)
    transport._async_agfs = storage._async_agfs
    return Session(viking_fs=transport, session_id="origin-test", session_uri=uri), storage


@pytest.mark.asyncio
async def test_ingress_stamp_precedes_build_and_readback_is_under_same_lock(
    origin_session, monkeypatch
):
    session, storage = origin_session
    events = []
    locked = False

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            events.append("clock")
            return cls(2026, 10, 17, tzinfo=timezone.utc)

    async def acquire(*args, **kwargs):
        nonlocal locked
        events.append("lock")
        locked = True
        return "own-test-lease"

    async def release(lease):
        nonlocal locked
        assert lease == "own-test-lease"
        locked = False
        events.append("release")

    read_file = storage.read_file

    async def read(uri, **kwargs):
        assert locked
        events.append("read")
        return await read_file(uri, **kwargs)

    monkeypatch.setattr("openviking.session.session.datetime", Clock)
    monkeypatch.setattr(storage, "read_file", read)
    storage._async_agfs.pathlock_acquire_exact.side_effect = acquire
    storage._async_agfs.pathlock_release.side_effect = release
    messages = await session.add_messages_async(
        [
            {
                "role": "user",
                "parts": [TextPart("synthetic")],
                "origin": source_statements(),
                "created_at": "2026-10-17T09:10:11.123456+07:00",
                "source_message_ids": ["record-A"],
            }
        ]
    )
    row = json.loads(storage.files[f"{session.uri}/messages.jsonl"])
    assert row["created_at"] == "2026-10-17T09:10:11.123456+07:00"
    assert row["origin"]["receiver"]["session_ingest_started_at"] == "2026-10-17T00:00:00+00:00"
    assert row["origin"]["contributors"] == source_statements()["contributors"]
    assert messages[0].to_dict() == row
    assert events.count("clock") == 1
    assert events[0:2] == ["clock", "lock"]
    assert events[-1] == "release"
    assert events.count("read") >= 3


@pytest.mark.asyncio
async def test_full_batch_rejection_precedes_any_storage_effect(origin_session):
    session, storage = origin_session
    with pytest.raises(ValueError):
        await session.add_messages_async(
            [
                {"role": "user", "parts": [TextPart("valid")], "origin": source_statements()},
                {"role": "user", "parts": [], "origin": {**source_statements(), "receiver": {}}},
            ]
        )
    assert storage.files == {}
    assert session.messages == []
    storage._async_agfs.pathlock_acquire_exact.assert_not_called()


@pytest.mark.asyncio
async def test_readback_detects_origin_loss_after_append(origin_session, monkeypatch):
    session, storage = origin_session
    write = storage.write_file

    async def lose_origin(uri, content, **kwargs):
        if uri.endswith("/messages.jsonl"):
            row = json.loads(content)
            row.pop("origin")
            content = json.dumps(row) + "\n"
        await write(uri, content, **kwargs)

    monkeypatch.setattr(storage, "write_file", lose_origin)
    with pytest.raises(OriginContractMismatch, match="readback mismatch"):
        await session.add_messages_async(
            [
                {
                    "role": "user",
                    "parts": [TextPart("persisted but uncertain")],
                    "origin": source_statements(),
                }
            ]
        )
    assert "persisted but uncertain" in storage.files[f"{session.uri}/messages.jsonl"]
    storage._async_agfs.pathlock_release.assert_awaited_once()


@pytest.mark.asyncio
async def test_tool_fanout_and_archive_preserve_input_group_candidates(origin_session):
    session, storage = origin_session
    messages = await session.add_messages_async(
        [
            {
                "role": "user",
                "parts": [ToolPart(tool_id="call"), ToolPart(tool_id="result")],
                "source_message_ids": ["call-row", "result-row"],
                "origin": source_statements(),
            }
        ]
    )
    scopes = [message.to_dict()["origin"]["receiver"]["applicability"] for message in messages]
    assert len(messages) == 2
    assert [scope["member_index"] for scope in scopes] == [0, 1]
    assert [scope["member_count"] for scope in scopes] == [2, 2]
    assert {scope["kind"] for scope in scopes} == {"input_group"}
    assert scopes[0]["group_id"] == scopes[1]["group_id"]
    archive = f"{session.uri}/history/archive_001"
    storage.files[f"{archive}/messages.jsonl"] = storage.files[f"{session.uri}/messages.jsonl"]
    restored = await session._archives.read_messages(archive)
    assert [m.to_dict()["origin"] for m in restored] == [m.to_dict()["origin"] for m in messages]


@pytest.mark.asyncio
async def test_new_legacy_omission_gets_only_known_ingest_start(origin_session):
    session, storage = origin_session
    message = await session.add_message_async("user", [TextPart("unmarked")])
    row = json.loads(storage.files[f"{session.uri}/messages.jsonl"])
    assert row["origin"]["version"] == 1
    assert row["origin"]["contributors"] == []
    assert row["origin"]["receiver"]["session_ingest_started_at"] == message.created_at


@pytest.mark.asyncio
async def test_explicit_origin_without_storage_fails_before_append(monkeypatch):
    session = Session(viking_fs=Mock(spec=VikingFS), session_id="no-storage")
    monkeypatch.setattr(session, "_viking_fs", None)
    with pytest.raises(OriginContractMismatch, match="requires native storage"):
        await session.add_message_async(
            "user", [TextPart("not accepted")], origin=_parse_origin_input(source_statements())
        )
    assert session.messages == []


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", [False, True])
async def test_http_origin_receipt_projects_all_actual_readback_members(
    origin_session, monkeypatch, batch
):
    from openviking.server.routers import sessions as routes

    session, storage = origin_session

    async def get_session(*args, **kwargs):
        return session

    service = SimpleNamespace(
        sessions=SimpleNamespace(
            get=get_session,
            maybe_schedule_auto_commit=AsyncMock(),
        )
    )
    monkeypatch.setattr(routes, "get_service", lambda: service)
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_session_request_context] = lambda: session.ctx
    message = {
        "role": "user",
        "parts": [{"type": "tool", "tool_id": "call"}, {"type": "tool", "tool_id": "result"}],
        "origin": source_statements(),
        "source_message_ids": ["call-row", "result-row"],
    }
    path = f"/api/v1/sessions/{session.session_id}/messages" + ("/batch" if batch else "")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://synthetic.invalid"
    ) as client:
        response = await client.post(path, json={"messages": [message]} if batch else message)
    assert response.status_code == 200
    result = response.json()["result"]
    stored = [
        json.loads(line) for line in storage.files[f"{session.uri}/messages.jsonl"].splitlines()
    ]
    assert result["origin_readback"] == "jsonl"
    assert result["message_count"] == 2
    if batch:
        assert result["added"] == 2
    assert [item["message_id"] for item in result["accepted_origins"]] == [
        row["id"] for row in stored
    ]
    assert [item["origin"] for item in result["accepted_origins"]] == [
        row["origin"] for row in stored
    ]
    assert [item["input_index"] for item in result["accepted_origins"]] == [0, 0]


@pytest.mark.parametrize(
    "extra",
    [
        {"receiver": {}},
        {"session_ingest_started_at": "1900-01-01T00:00:00Z"},
        {"origin": {**source_statements(), "receiver": {}}},
    ],
)
def test_http_request_type_rejects_receiver_injection(extra):
    from openviking.server.routers.sessions import AddMessageRequest

    with pytest.raises(ValueError):
        AddMessageRequest.model_validate({"role": "user", "content": "untrusted", **extra})
