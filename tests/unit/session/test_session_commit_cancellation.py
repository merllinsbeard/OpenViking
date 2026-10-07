# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Isolated control unit checks with in-memory storage and mocked model/transport IO."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.message import Message, TextPart
from openviking.pyagfs import AsyncAGFSClient
from openviking.service.task_queue_middleware import TaskWorkQueueMiddleware
from openviking.service.task_tracker import TaskStatus, TaskTracker
from openviking.service.task_work_index import bind_task_context
from openviking.session.session import Session
from openviking.storage.queuefs.named_queue import DequeueHandlerBase, NamedQueue
from openviking.storage.queuefs.process_result import ProcessResult
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


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("during_start", [False, True])
async def test_cancellation_only_marks_explicit_request(
    commit_run, monkeypatch, explicit, during_start
):
    session, storage, tracker, job = commit_run
    await tracker.create(
        "session_commit", task_id=job.task_id, account_id="default", user_id="default"
    )
    if explicit:
        await tracker.cancel(job.task_id, account_id="default", user_id="default")
    target = "start" if during_start else "_prepare_phase2_archive_messages"
    monkeypatch.setattr(
        tracker if during_start else session, target, AsyncMock(side_effect=asyncio.CancelledError)
    )

    with pytest.raises(asyncio.CancelledError):
        await session.resume_queued_commit(job)

    marker = storage.files.get(f"{job.archive_uri}/.failed.json")
    if explicit:
        assert json.loads(marker)["stage"] == "cancelled"
        assert json.loads(marker)["error"] == "session commit cancelled"
    else:
        assert marker is None
        assert (await tracker.get(job.task_id)).status is TaskStatus.PENDING
    assert f"{job.archive_uri}/.done" not in storage.files


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

    assert (await tracker.get(job.task_id)).status is TaskStatus.RUNNING
    assert f"{job.archive_uri}/.failed.json" not in storage.files
    assert f"{job.archive_uri}/.done" not in storage.files


async def test_shutdown_delivery_is_unacked_and_reuses_persisted_progress(commit_run, monkeypatch):
    session, storage, tracker, job = commit_run
    archived_content = storage.files[f"{job.archive_uri}/messages.jsonl"]
    progress_saved = asyncio.Event()
    original_merge = session._merge_archive_meta

    async def merge(uri, updates, **kwargs):
        await original_merge(uri, updates, **kwargs)
        if "completed_memory_steps" in updates:
            progress_saved.set()

    async def blocked_summary(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(session, "_merge_archive_meta", merge)
    session._generate_archive_summary_async.side_effect = blocked_summary
    transport = AsyncMock(spec=AsyncAGFSClient)
    transport.write.return_value = "delivery-1"
    monkeypatch.setattr(
        "openviking.storage.queuefs.named_queue.AsyncAGFSClient", lambda _: transport
    )

    class Handler(DequeueHandlerBase):
        async def on_dequeue(self, data):
            assert await session.resume_queued_commit(job)
            return ProcessResult.success(data)

    queue = NamedQueue(
        object(),
        "/queue",
        "SessionCommit",
        dequeue_handler=Handler(),
        middlewares=[TaskWorkQueueMiddleware(tracker._work_index)],
    )
    with bind_task_context(job.task_id, "default", "default"):
        await queue.enqueue(job.to_dict())
    delivery = {"id": "delivery-1", "data": transport.write.await_args.args[1].decode()}
    transport.read.return_value = json.dumps(delivery).encode()
    transport.write.reset_mock()
    worker = asyncio.create_task(queue.dequeue())
    try:
        await asyncio.wait_for(progress_saved.wait(), timeout=3)
    finally:
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker

    assert (await tracker.get(job.task_id)).status is TaskStatus.RUNNING
    assert tracker.has_work(job.task_id)
    assert json.loads(storage.files[f"{job.archive_uri}/.meta.json"])["completed_memory_steps"] == {
        "long_term": ["m1"]
    }
    assert f"{job.archive_uri}/.failed.json" not in storage.files
    assert f"{job.archive_uri}/.done" not in storage.files
    transport.write.assert_not_awaited()

    previous = session
    session = Session(viking_fs=storage, session_id=job.session_id, session_uri=job.session_uri)
    session._session_compressor = previous._session_compressor
    monkeypatch.setattr(session, "_run_usage_reporting", previous._run_usage_reporting)
    monkeypatch.setattr(
        session._checkpoints,
        "collect_requests_for_phase2",
        previous._checkpoints.collect_requests_for_phase2,
    )
    monkeypatch.setattr(
        session._archives,
        "latest_completed_overview",
        previous._archives.latest_completed_overview,
    )
    monkeypatch.setattr(session, "_generate_archive_summary_async", AsyncMock(return_value=""))
    monkeypatch.setattr(
        session, "_merge_and_save_commit_meta", previous._merge_and_save_commit_meta
    )
    assert await queue.dequeue() == delivery
    assert (await tracker.get(job.task_id)).status is TaskStatus.COMPLETED
    assert json.loads(storage.files[f"{job.archive_uri}/.done"])["completed_memory_steps"] == {
        "long_term": ["m1"]
    }
    assert session._session_compressor.extract_long_term_memories.await_count == 1
    assert not tracker.has_work(job.task_id)
    meta = json.loads(storage.files[f"{job.archive_uri}/.meta.json"])
    assert meta["phase1"] == {"status": "ready"}
    assert meta["checkpoints"] == [{"id": "synthetic-checkpoint"}]
    assert storage.files[f"{job.archive_uri}/messages.jsonl"] == archived_content
    transport.write.assert_awaited_once_with("/queue/SessionCommit/ack", b"delivery-1")
