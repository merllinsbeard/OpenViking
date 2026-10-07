# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Isolated native filesystem lock checks with deterministic thread barriers."""

import asyncio
import threading

import pytest

from openviking.pyagfs import AsyncAGFSClient, get_binding_client

CTX = {"account_id": "default"}
PATH = "/local/default/synthetic"


@pytest.fixture
def native_client(tmp_path):
    try:
        binding, _ = get_binding_client()
    except ImportError:
        pytest.skip("ragfs_python native binding is unavailable")
    client = binding(
        config={
            "pathlock": {"provider": "filesystem", "lock_expire_secs": 30},
            "cache": {"enabled": False, "runtime_enabled": False},
        }
    )
    client.mount("localfs", "/local", {"local_dir": str(tmp_path)})
    try:
        yield client
    finally:
        client.close()


class DelayedAcquisition:
    def __init__(self, client, method):
        self.client = client
        self.method = method
        self.started = threading.Event()
        self.return_allowed = threading.Event()
        self.release_started = threading.Event()
        self.release_allowed = threading.Event()
        self.release_allowed.set()

    def __getattr__(self, name):
        native = getattr(self.client, name)
        if name != self.method:
            return native

        def acquire(*args):
            lease = native(*args)
            self.started.set()
            if not self.return_allowed.wait(3):
                raise TimeoutError("test acquisition barrier was not released")
            return lease

        return acquire

    def pathlock_release(self, *args):
        self.release_started.set()
        if not self.release_allowed.wait(3):
            raise TimeoutError("test release barrier was not released")
        return self.client.pathlock_release(*args)


ACQUISITIONS = [
    ("pathlock_acquire_exact", [PATH]),
    ("pathlock_acquire_exact_batch", [[PATH]]),
    ("pathlock_acquire_tree", [PATH]),
    ("pathlock_acquire_tree_batch", [[PATH]]),
    ("pathlock_acquire_exact_tree_batch", [[], [PATH]]),
    ("pathlock_acquire_batch", [[{"path": PATH, "kind": "tree"}]]),
]


@pytest.mark.parametrize("method,args", ACQUISITIONS)
async def test_cancelled_acquisition_releases_native_lease(native_client, method, args):
    delayed = DelayedAcquisition(native_client, method)
    client = AsyncAGFSClient(delayed)
    operation = asyncio.create_task(getattr(client, method)(*args, fs_ctx=CTX))
    try:
        assert await asyncio.to_thread(delayed.started.wait, 3)
        operation.cancel()
        await asyncio.sleep(0)
    finally:
        delayed.return_allowed.set()
        with pytest.raises(asyncio.CancelledError):
            await operation

    assert native_client.pathlock_observe(CTX)["active_locks"] == 0
    lease = native_client.pathlock_acquire_exact(CTX, PATH)
    native_client.pathlock_release(CTX, lease)


async def test_repeated_cancellation_waits_for_native_release(native_client):
    delayed = DelayedAcquisition(native_client, "pathlock_acquire_tree")
    delayed.release_allowed.clear()
    client = AsyncAGFSClient(delayed)
    operation = asyncio.create_task(client.pathlock_acquire_tree(PATH, fs_ctx=CTX))
    try:
        assert await asyncio.to_thread(delayed.started.wait, 3)
        operation.cancel()
        delayed.return_allowed.set()
        assert await asyncio.to_thread(delayed.release_started.wait, 3)
        operation.cancel()
        await asyncio.sleep(0)
        assert not operation.done()
    finally:
        delayed.return_allowed.set()
        delayed.release_allowed.set()
        with pytest.raises(asyncio.CancelledError):
            await operation
    assert native_client.pathlock_observe(CTX)["active_locks"] == 0


async def test_successful_acquisition_remains_owned_until_caller_releases(native_client):
    client = AsyncAGFSClient(native_client)
    lease = await client.pathlock_acquire_tree(PATH, fs_ctx=CTX)
    assert native_client.pathlock_observe(CTX)["active_locks"] == 1
    await client.pathlock_release(lease, fs_ctx=CTX)
    assert native_client.pathlock_observe(CTX)["active_locks"] == 0


async def test_cancelled_child_acquisition_preserves_outer_owner(native_client):
    parent = native_client.pathlock_acquire_tree(CTX, PATH)
    delayed = DelayedAcquisition(native_client, "pathlock_acquire_exact")
    client = AsyncAGFSClient(delayed)
    operation = asyncio.create_task(
        client.pathlock_acquire_exact(f"{PATH}/child", owner_lease_ref=parent, fs_ctx=CTX)
    )
    try:
        assert await asyncio.to_thread(delayed.started.wait, 3)
        operation.cancel()
        await asyncio.sleep(0)
    finally:
        delayed.return_allowed.set()
        with pytest.raises(asyncio.CancelledError):
            await operation
    assert native_client.pathlock_is_locked(CTX, f"{PATH}/child") is True
    native_client.pathlock_release(CTX, parent)
    assert native_client.pathlock_observe(CTX)["active_locks"] == 0


async def test_failed_acquisition_preserves_conflicting_native_owner(native_client):
    parent = native_client.pathlock_acquire_tree(CTX, PATH)
    client = AsyncAGFSClient(native_client)
    with pytest.raises(Exception, match="lock acquire"):
        await client.pathlock_acquire_exact(f"{PATH}/child", fs_ctx=CTX)
    assert native_client.pathlock_observe(CTX)["active_locks"] == 1
    native_client.pathlock_release(CTX, parent)
    assert native_client.pathlock_observe(CTX)["active_locks"] == 0
