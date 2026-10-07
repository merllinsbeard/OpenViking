# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Real generated ZIP compression; config resolution uses an in-memory test double."""

import asyncio
import json
import threading
import zipfile
from array import array
from types import SimpleNamespace

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.ovpack.format import dense_values_bytes
from openviking.storage.ovpack.index import build_manifest
from openviking.storage.ovpack.operations import _write_ovpack_archive
from openviking.storage.ovpack.validation import validate_manifest_content
from openviking_cli.session.user_id import UserIdentifier


async def _write_generated(path, values):
    class GeneratedResolver:
        async def resolve(self, account_id):
            return SimpleNamespace(embedding=SimpleNamespace(hybrid=None, dense=None))

    manifest, _, _ = await build_manifest(None, None, "viking://", "generated", [], None)
    records = [
        {
            "record_id": str(i),
            "path": "",
            "kind": "directory",
            "level": 2,
            "vector": {"dense": {"offset": i * 4096, "dimensions": 4096}},
        }
        for i in range(len(values) // 4096)
    ]
    ctx = RequestContext(UserIdentifier("generated", "generated"), Role.ADMIN)
    await _write_ovpack_archive(
        None,
        "viking://",
        str(path),
        "generated",
        [],
        manifest,
        records,
        values,
        ctx,
        None,
        GeneratedResolver(),
    )
    return records


def test_real_dense_zip_compression_keeps_event_loop_running(tmp_path, monkeypatch):
    values = array("f", ((i * 2654435761 & 0xFFFFFFFF) / 2**32 for i in range(262144)))
    path = tmp_path / "generated.ovpack"
    active = threading.Event()
    finished = threading.Event()
    writer_threads = []
    original = zipfile.ZipFile.writestr

    def observed_write(zf, name, data, *args, **kwargs):
        if not name.endswith("/dense.f32"):
            return original(zf, name, data, *args, **kwargs)
        writer_threads.append(threading.get_ident())
        active.set()
        try:
            return original(zf, name, data, *args, **kwargs)
        finally:
            active.clear()
            finished.set()

    monkeypatch.setattr(zipfile.ZipFile, "writestr", observed_write)

    async def run():
        heartbeats = 0

        async def heartbeat():
            nonlocal heartbeats
            while not finished.is_set():
                if active.is_set():
                    heartbeats += 1
                await asyncio.sleep(0.001)

        ticker = asyncio.create_task(heartbeat())
        try:
            records = await _write_generated(path, values)
        finally:
            finished.set()
            await ticker
        assert heartbeats >= 2
        return records

    records = asyncio.run(run())
    assert writer_threads == [writer_threads[0]]
    assert writer_threads[0] != threading.get_ident()
    with zipfile.ZipFile(path) as zf:
        assert zf.namelist() == [
            "generated/",
            "generated/files/",
            "generated/_ovpack/",
            "generated/_ovpack/index_records.jsonl",
            "generated/_ovpack/dense.f32",
            "generated/_ovpack/manifest.json",
        ]
        assert zf.read("generated/_ovpack/dense.f32") == dense_values_bytes(values)
        manifest = json.loads(zf.read("generated/_ovpack/manifest.json"))
        assert manifest["index"]["dense"]["count"] == 64
        assert manifest["index"]["dense"]["dimensions"] == 4096
        assert validate_manifest_content(zf, manifest, zf.infolist(), "generated") == records
        assert zf.testzip() is None


def test_cancel_waits_for_zip_worker_before_closing(tmp_path, monkeypatch):
    """The barrier injects slow IO; the worker still calls the real ZIP writer."""
    path = tmp_path / "cancelled.ovpack"
    entered, release = threading.Event(), threading.Event()
    observed = []
    original = zipfile.ZipFile.writestr

    def delayed_write(zf, name, data, *args, **kwargs):
        if name.endswith("/dense.f32"):
            entered.set()
            assert release.wait(5), "test barrier was not released"
            observed.append(zf.fp is not None)
        return original(zf, name, data, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "writestr", delayed_write)

    async def run():
        task = asyncio.create_task(_write_generated(path, array("f", [1.0]) * 4096))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0.01)
                assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert observed == [True]
    with zipfile.ZipFile(path) as zf:
        assert zf.read("generated/_ovpack/dense.f32") == bytes.fromhex("0000803f") * 4096
        assert "generated/_ovpack/manifest.json" not in zf.namelist()
        assert zf.testzip() is None


def test_dense_zip_worker_errors_propagate_and_close(tmp_path, monkeypatch):
    path = tmp_path / "failed.ovpack"
    original = zipfile.ZipFile.writestr

    def failing_write(zf, name, data, *args, **kwargs):
        if name.endswith("/dense.f32"):
            raise OSError("generated IO failure")
        return original(zf, name, data, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "writestr", failing_write)
    with pytest.raises(OSError, match="generated IO failure"):
        asyncio.run(_write_generated(path, array("f", [1.0]) * 4096))
    with zipfile.ZipFile(path) as zf:
        assert "generated/_ovpack/manifest.json" not in zf.namelist()
        assert zf.testzip() is None
