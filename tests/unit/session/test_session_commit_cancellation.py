# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Isolated control unit checks with in-memory storage and mocked model/transport IO."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.message import Message, TextPart
from openviking.service.task_tracker import TaskStatus, TaskTracker
from openviking.session.session import Session
from openviking.storage.queuefs.session_commit_msg import SessionCommitMsg
from tests.unit.session.test_session_commit_resume import _MemoryVikingFS, _TaskStore


@pytest.fixture
def commit_run(monkeypatch):
    session_uri = "viking://user/default/sessions/session-1"
    archive_uri = f"{session_uri}/history/archive_001"
    message = Message(id="m1", role="user", parts=[TextPart("synthetic test message")])
    storage = _MemoryVikingFS(
        {
            f"{archive_uri}/messages.jsonl": message.to_jsonl(),
            f"{archive_uri}/.meta.json": json.dumps(
                {
                    "phase1": {"status": "ready"},
                    "checkpoints": [{"id": "synthetic-checkpoint"}],
                }
            ),
        }
    )
    tracker = TaskTracker(_TaskStore())
    monkeypatch.setattr("openviking.service.task_tracker.get_task_tracker", lambda: tracker)
    monkeypatch.setattr(
        "openviking.session.session.get_openviking_config",
        lambda: SimpleNamespace(
            memory=SimpleNamespace(extraction_enabled=True, session_skill_extraction_enabled=False)
        ),
    )
    session = Session(viking_fs=storage, session_id="session-1", session_uri=session_uri)
    session._session_compressor = SimpleNamespace(
        extract_long_term_memories=AsyncMock(return_value={"contexts": []})
    )
    monkeypatch.setattr(session, "_run_usage_reporting", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        session._checkpoints, "collect_requests_for_phase2", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(session._archives, "latest_completed_overview", AsyncMock(return_value=""))
    monkeypatch.setattr(session, "_generate_archive_summary_async", AsyncMock(return_value=""))
    monkeypatch.setattr(session, "_merge_and_save_commit_meta", AsyncMock())
    job = SessionCommitMsg(
        task_id="task-1",
        session_id="session-1",
        session_uri=session_uri,
        archive_uri=archive_uri,
        user={"account_id": "default", "user_id": "default"},
        memory_policy={"memory_types": ["profile"]},
    )
    return session, storage, tracker, job


@pytest.mark.parametrize("sibling_fails", [False, True])
@pytest.mark.parametrize("cancelled_step", ["archive_summary", "long_term"])
async def test_child_cancellation_propagates_without_terminal_marker(
    commit_run, sibling_fails, cancelled_step
):
    session, storage, tracker, job = commit_run
    steps = {
        "archive_summary": session._generate_archive_summary_async,
        "long_term": session._session_compressor.extract_long_term_memories,
    }
    for name, step in steps.items():
        if name == cancelled_step:
            step.side_effect = asyncio.CancelledError
        elif sibling_fails:
            step.side_effect = ValueError("synthetic extraction failure")

    with pytest.raises(asyncio.CancelledError):
        await session.resume_queued_commit(job)

    task = await tracker.get(job.task_id, account_id="default", user_id="default")
    assert task is not None
    assert task.status is TaskStatus.RUNNING
    assert f"{job.archive_uri}/.failed.json" not in storage.files
    assert f"{job.archive_uri}/.done" not in storage.files

