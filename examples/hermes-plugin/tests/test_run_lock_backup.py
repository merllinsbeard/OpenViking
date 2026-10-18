"""Real POSIX locks and native archive reads; no server, CLI or network calls."""
import socket
import zipfile

import pytest


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("run-lock tests must not connect to a server")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)


def test_released_run_lock_preserves_inode(external_provider):
    home, provider, module, _ = external_provider("stable-lock")
    if module.fcntl is None:
        pytest.skip("requires native POSIX flock")
    provider._hermes_home = str(home)
    provider._acquire_run_lock()
    path = provider._run_lock_path
    inode = path.stat().st_ino
    provider._release_run_lock()
    assert path.stat().st_ino == inode
    handle = provider._flock_open(path)
    try:
        assert path.stat().st_ino == inode
    finally:
        provider._flock_close(handle, path, "test lock")


def test_busy_run_lock_does_not_unlink_owner_inode(external_provider):
    home, owner, module, _ = external_provider("busy-lock")
    if module.fcntl is None:
        pytest.skip("requires native POSIX flock")
    _, contender, _, _ = external_provider("busy-lock")
    owner._hermes_home = contender._hermes_home = str(home)
    contender._run_id = owner._run_id
    owner._acquire_run_lock()
    path = owner._run_lock_path
    inode = path.stat().st_ino
    try:
        contender._acquire_run_lock()
        assert contender._run_lock_file is None
        assert path.stat().st_ino == inode
        with pytest.raises(BlockingIOError):
            owner._flock_open(path)
    finally:
        owner._release_run_lock()
        contender._release_run_lock()


@pytest.mark.parametrize("prefix", ["", "profiles/nested"])
def test_backup_survives_run_lock_release_after_inventory(external_provider, tmp_path, prefix):
    from hermes_cli import backup

    _, provider, module, _ = external_provider("archive-lock")
    if module.fcntl is None:
        pytest.skip("requires native POSIX flock")
    root = tmp_path / "archive-home"
    profile = root / prefix
    provider._hermes_home = str(profile)
    provider._acquire_run_lock()
    lock = provider._run_lock_path
    marker = profile / "openviking/pending_sessions/synthetic.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = b'{"synthetic_recovery_state":true}\n'
    marker.write_bytes(payload)
    output = tmp_path / "native.zip"
    entries = list(backup._iter_backup_files(root, output))
    assert any(path == lock for path, _ in entries)
    provider._release_run_lock()
    errors = []
    with zipfile.ZipFile(output, "w") as archive:
        backup._write_zip_entries(archive, entries, output,
            on_db_failure=lambda path: errors.append((path, "database failure")),
            on_error=lambda path, error: errors.append((path, type(error).__name__)),
            on_progress=lambda _: None, track_bytes=True)
    assert errors == []
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert archive.read(str(lock.relative_to(root))) == b""
        assert archive.read(str(marker.relative_to(root))) == payload
