"""Load the plugin through Hermes discovery in isolated profile homes."""

import copy
import importlib.util
import os
import shutil
import sys
from pathlib import Path

import pytest


@pytest.fixture
def origin_ack():
    """MOCK HTTP receipt; the plugin sender and its validation stay native."""
    def response(messages):
        members = []
        stamp = "2026-10-17T00:00:00Z"
        for index, message in enumerate(messages):
            parts = message.get("parts") or []
            count = len(parts) if message.get("role") == "user" and len(parts) > 1 and all(part.get("type") == "tool" for part in parts) else 1
            for slot in range(count):
                scope = {"kind": "input_message", "group_id": f"mock-group-{index}"} if count == 1 else {"kind": "input_group", "group_id": f"mock-group-{index}", "member_index": slot, "member_count": count}
                origin = copy.deepcopy(message.get("origin") or {"version": 1, "contributors": []})
                origin["receiver"] = {"session_ingest_started_at": stamp, "applicability": scope}
                members.append({"input_index": index, "message_id": f"mock-{index}-{slot}", "origin": origin,
                                "legacy_created_at": message.get("created_at") or stamp, "turn_id": message.get("turn_id"),
                                "message_kind": message.get("message_kind") or ("tool_transport" if count > 1 else None),
                                "source_message_ids": message.get("source_message_ids")})
        return {"result": {"origin_readback": "jsonl", "accepted_origins": members}}
    return response


@pytest.fixture
def source_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key in list(os.environ):
        if key.startswith("OPENVIKING_"):
            monkeypatch.delenv(key)
    name = "openviking_source_test"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / "__init__.py",
        submodule_search_locations=[str(Path(__file__).resolve().parents[1])],
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    provider = module.OpenVikingMemoryProvider()
    provider._hermes_home = str(tmp_path)
    provider._hermes_home_bound = True
    (tmp_path / "config.yaml").write_text(
        "memory:\n  openviking:\n    use_ovcli_config: false\n", encoding="utf-8"
    )
    yield provider, module
    provider._writes_enabled = False
    provider.shutdown()
    for key in list(sys.modules):
        if key.startswith(name + "."):
            del sys.modules[key]


@pytest.fixture
def external_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
    for key in list(os.environ):
        if key.startswith("OPENVIKING_"):
            monkeypatch.delenv(key)

    import plugins.memory as memory
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setattr(memory, "_MEMORY_PLUGINS_DIR", tmp_path / "empty-bundled")
    providers = []

    def load(profile):
        home = tmp_path / profile
        target = home / "plugins" / "openviking"
        if not target.exists():
            shutil.copytree(
                Path(__file__).resolve().parents[1],
                target,
                ignore=shutil.ignore_patterns("tests", "__pycache__", ".pytest_cache"),
            )
            (home / "config.yaml").write_text(
                "memory:\n  provider: openviking\n  openviking:\n"
                "    use_ovcli_config: false\n"
                f"    agent: {profile}\n",
                encoding="utf-8",
            )
        token = set_hermes_home_override(home)
        try:
            assert memory.find_provider_dir("openviking") == target
            provider = memory.load_memory_provider("openviking", register_skills=False)
            assert provider is not None
            module = sys.modules[type(provider).__module__]
            settings = module._resolve_connection_settings(module._load_hermes_openviking_config())
        finally:
            reset_hermes_home_override(token)
        providers.append(provider)
        return home, provider, module, settings

    yield load
    for provider in providers:
        provider.shutdown()
