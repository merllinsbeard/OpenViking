
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from hermes_cli.config import yaml


@contextmanager
def recall_server(respond):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            respond(self)

        def do_POST(self):
            self.payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            respond(self)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def send(handler, result, *, delay=0, status=200, raw=False):
    time.sleep(delay)
    payload = {"error": {"message": "Trusted mode requests must include X-OpenViking-Account"}} if status == 400 else {"result": result}
    if raw:
        payload = result
    body = json.dumps(payload).encode()
    try:
        handler.send_response(status)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError):
        pass


def configure(external_provider, endpoint, **settings):
    from agent.memory_manager import MemoryManager

    home, provider, module, _ = external_provider("deadline")
    cfg = {"endpoint": endpoint, "use_ovcli_config": False, "recall_mode": "find",
           "recall_timeout_seconds": 0.6, "recall_request_timeout_seconds": 0.5,
           "recall_prefer_abstract": True, **settings}
    (home / "config.yaml").write_text(yaml.safe_dump({"memory": {"provider": "openviking", "openviking": cfg}}))
    provider._hermes_home = str(home)
    provider._hermes_home_bound = True
    provider._endpoint = endpoint
    provider._client = module._VikingClient(endpoint, "test-key", account="test", user="test", agent="")
    provider._session_id = "deadline-session"
    manager = MemoryManager(external_prefetch_timeout=0.9)
    manager.add_provider(provider)
    return provider, module, manager


def test_query_survives_startup_budget_exhaustion_and_next_turn_recovers(external_provider):
    paths = []
    delay_query = [True]

    def respond(handler):
        paths.append(handler.path)
        if handler.path == "/api/v1/system/status":
            send(handler, {"user": "test"}, delay=0.08)
        elif handler.path.startswith("/api/v1/content/read"):
            send(handler, {"content": "Operator prefers source citations."}, delay=0.08)
        elif handler.path.startswith("/api/v1/fs/ls"):
            send(handler, [], delay=0.08)
        else:
            send(handler, {"memories": [{"uri": "viking://user/test/memories/decision.md",
                                        "abstract": "Use the native source first.", "score": 0.9}]},
                 delay=0.4 if delay_query[0] else 0)

    with recall_server(respond) as endpoint:
        provider, _, manager = configure(external_provider, endpoint)
        started = time.monotonic()
        block = manager.prefetch_all("Which source should I use?", session_id="deadline-session")
        elapsed = time.monotonic() - started
        assert "Use the native source first." in block
        assert "viking://user/test/memories/decision.md" in block
        assert elapsed < 0.85
        assert not manager._external_prefetch_threads
        delay_query[0] = False
        block = manager.prefetch_all("Which source should I use?", session_id="deadline-session")
        assert "Use the native source first." in block
        assert "viking://user/test/memories/decision.md" in block
        assert paths.count("/api/v1/search/find") == 2
        assert "/api/v1/search/search" not in paths


def test_trickled_body_is_cancelled_without_retry_or_stuck_prefetch(external_provider):
    disconnected = threading.Event()
    paths = []

    def respond(handler):
        paths.append(handler.path)
        handler.send_response(200)
        handler.send_header("Content-Length", "1000")
        handler.end_headers()
        try:
            for _ in range(40):
                handler.wfile.write(b" ")
                handler.wfile.flush()
                time.sleep(0.04)
        except (BrokenPipeError, ConnectionResetError):
            disconnected.set()

    with recall_server(respond) as endpoint:
        provider, _, manager = configure(external_provider, endpoint, recall_request_timeout_seconds=0.3)
        provider._profile_prefetched_sessions.add("deadline-session")
        started = time.monotonic()
        assert manager.prefetch_all("Recall the source decision", session_id="deadline-session") == ""
        assert time.monotonic() - started < 0.6
        assert disconnected.wait(0.5)
        assert not manager._external_prefetch_threads
        assert paths == ["/api/v1/search/find"]


@pytest.mark.parametrize("refresh", [False, True])
def test_identity_retry_and_refresh_lock_cannot_restart_budget(external_provider, refresh):
    paths = []

    def respond(handler):
        paths.append(handler.path)
        if len(paths) == 1:
            send(handler, {}, delay=0.35, status=400)
        else:
            send(handler, {}, delay=0.35)

    with recall_server(respond) as endpoint:
        provider, _, manager = configure(external_provider, endpoint)
        if refresh:
            provider._env_refresh_enabled = True
            provider._client_refresh_lock.acquire()
        try:
            started = time.monotonic()
            assert manager.prefetch_all("Recall source decisions", session_id="deadline-session") == ""
            assert time.monotonic() - started < 0.85
            assert not manager._external_prefetch_threads
            assert len(paths) == (0 if refresh else 2)
        finally:
            if refresh:
                provider._client_refresh_lock.release()


def test_native_background_worker_does_not_inherit_recall_transport(external_provider):
    from agent.memory_provider import spawn_context_thread

    def respond(handler):
        send(handler, {"content": "The captured source remains available."}, delay=0.2)

    with recall_server(respond) as endpoint:
        provider, module, _ = configure(external_provider, endpoint)
        budget = module._RecallBudget(time.monotonic() + 0.1, 0.05)
        token = module._recall_budget.set(budget)
        result = []
        worker = spawn_context_thread(
            lambda: result.append(provider._client.get("/api/v1/content/read")), name="capture-check"
        )
        try:
            worker.start()
        finally:
            module._recall_budget.reset(token)
            budget.close()
        worker.join(1)
        assert result == [{"result": {"content": "The captured source remains available."}}]


@pytest.mark.parametrize("change,fault_path,request_timeout", [
    ("endpoint", "/health", 0.15),
    ("identity", "/health", 1.0),
    ("api-key", "/openapi.json", 0.15),
])
def test_refresh_timeout_retires_old_connection_and_recovers_after_cooldown(
    external_provider, monkeypatch, change, fault_path, request_timeout,
):
    from agent.memory_manager import MemoryManager

    calls = []
    fault = threading.Event()
    disconnected = threading.Event()

    def respond(label):
        def handle(handler):
            identity = (handler.headers.get("X-API-Key"), handler.headers.get("X-OpenViking-Account"),
                        handler.headers.get("X-OpenViking-User"), handler.headers.get("X-OpenViking-Actor-Peer"))
            calls.append((label, handler.path, identity))
            if fault.is_set() and handler.path == fault_path:
                handler.send_response(200)
                handler.send_header("Content-Length", "1000")
                handler.end_headers()
                try:
                    for _ in range(100):
                        handler.wfile.write(b" ")
                        handler.wfile.flush()
                        time.sleep(0.02)
                except (BrokenPipeError, ConnectionResetError):
                    disconnected.set()
            elif handler.path == "/health":
                health = {"status": "ok"}
                if fault_path != "/openapi.json":
                    health.update(healthy=True, version="fixture")
                send(handler, health, raw=True)
            elif handler.path == "/openapi.json":
                send(handler, {"info": {"title": "OpenViking API"}}, raw=True)
            elif handler.path == "/api/v1/system/status":
                send(handler, {"user": handler.headers.get("X-OpenViking-User") or "user-b"})
            elif handler.path == "/api/v1/search/find":
                peer = handler.headers.get("X-OpenViking-Actor-Peer")
                send(handler, {"memories": [{"uri": "viking://user/user-b/memories/decision.md",
                                            "abstract": f"Source for {peer}.", "score": 0.9}]})
            elif handler.path.startswith("/api/v1/content/"):
                send(handler, {"content": ""})
            elif handler.path.startswith("/api/v1/sessions/"):
                send(handler, {"pending_tokens": 0})
            else:
                send(handler, [])
        return handle

    with recall_server(respond("endpoint-a")) as endpoint_a, recall_server(respond("endpoint-b")) as endpoint_b:
        home, provider, module, _ = external_provider("refresh-fault-fixture")
        monkeypatch.setenv("HERMES_HOME", str(home))
        config = {"endpoint": endpoint_a, "use_ovcli_config": False, "agent": "peer-a",
                  "account": "account-a", "user": "user-a", "recall_mode": "find",
                  "recall_timeout_seconds": 0.4, "recall_request_timeout_seconds": request_timeout,
                  "recall_prefer_abstract": True}

        def save():
            (home / "config.yaml").write_text(yaml.safe_dump({"memory": {"provider": "openviking", "openviking": config}}))

        save()
        manager = MemoryManager(external_prefetch_timeout=0.9)
        manager.add_provider(provider)
        manager.initialize_all("refresh-session", hermes_home=str(home))
        assert "Source for peer-a." in manager.prefetch_all("Which source?", session_id="refresh-session")
        config.update(agent="peer-b", account="account-b", user="user-b")
        if change == "endpoint":
            config["endpoint"] = endpoint_b
        elif change == "api-key":
            (home / ".env").write_text("OPENVIKING_API_KEY=fixture-key-b\n")
        save()
        monkeypatch.setattr(module, "_FAILED_CONFIG_RETRY_COOLDOWN_SECONDS", 0.5)
        calls.clear()
        fault.set()
        started = time.monotonic()
        assert manager.prefetch_all("Which source?", session_id="refresh-session") == ""
        assert time.monotonic() - started < 0.8
        assert not manager._external_prefetch_threads
        assert disconnected.wait(0.2)
        assert not provider._client_refresh_lock.locked()

        fault.clear()
        failed_calls = list(calls)
        tool = json.loads(provider.handle_tool_call("viking_search", {"query": "Which source?"}))
        assert tool == {"error": "OpenViking server not connected"}
        provider.sync_turn("Source requested during cooldown", "No connection", session_id="refresh-session")
        assert manager.prefetch_all("Which source?", session_id="refresh-session") == ""
        assert calls == failed_calls

        time.sleep(0.51)
        block = manager.prefetch_all("Which source?", session_id="refresh-session")
        assert "Source for peer-b." in block
        assert "Source for peer-a." not in block
        assert not manager._external_prefetch_threads
        assert not provider._client_refresh_lock.locked()
        assert json.loads(provider.handle_tool_call("viking_search", {"query": "Which source?"}))["results"][0]["abstract"] == "Source for peer-b."
        provider.sync_turn("Source requested after recovery", "Connected to B", session_id="refresh-session")
        assert provider._drain_writers("refresh-session", timeout=1)
        recovered_calls = calls[len(failed_calls):]
        target = "endpoint-b" if change == "endpoint" else "endpoint-a"
        expected_identity = ("fixture-key-b", None, None, "peer-b") if change == "api-key" else (
            None, "account-b", "user-b", "peer-b",
        )
        assert all(label == target for label, _, _ in calls)
        assert all(identity == expected_identity for _, path, identity in recovered_calls if path.startswith("/api/"))
        assert any(path.endswith("/messages/batch") for _, path, _ in recovered_calls)
