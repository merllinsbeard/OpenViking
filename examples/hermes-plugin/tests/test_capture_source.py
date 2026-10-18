import copy
import json

import httpx
import pytest


def test_supplied_source_fields_reach_batch_without_author_inference(source_provider):
    provider, _ = source_provider
    source = [{
        "role": "user", "content": "Quoted material",
        "timestamp": "2026-10-17T22:15:16.123+07:00", "turn_id": "native-turn",
        "message_kind": "checkpoint", "source_message_ids": ["source-7", "source-7"],
        "message_uid": "durable-user", "_absorbed_message_uids": ["earlier-user"],
        "platform_message_id": "telegram-42", "author": "unverified label",
    }, {"role": "assistant", "content": "Reply", "created_at": "2026-10-17T15:15:17Z"}]
    original = copy.deepcopy(source)
    assert provider._messages_to_openviking_batch(source, user_peer_id="sender") == [{
        "role": "user", "parts": [{"type": "text", "text": "Quoted material"}],
        "peer_id": "sender", "created_at": "2026-10-17T22:15:16.123+07:00",
        "turn_id": "native-turn", "message_kind": "checkpoint",
        "source_message_ids": ["source-7", "durable-user", "earlier-user", "telegram-42"],
    }, {
        "role": "assistant", "parts": [{"type": "text", "text": "Reply"}],
        "created_at": "2026-10-17T15:15:17Z",
    }]
    assert source == original


@pytest.mark.parametrize(("timestamp", "expected"), [
    (0, "1970-01-01T00:00:00+00:00"),
    (1.25, "1970-01-01T00:00:01.250000+00:00"),
    (-1, "1969-12-31T23:59:59+00:00"),
    ("2026-10-17T22:15:16+07:00", "2026-10-17T22:15:16+07:00"),
    (None, None), (True, None), (float("nan"), None),
    (float("inf"), None), (10**30, None),
])
def test_timestamp_is_supplied_string_or_epoch_seconds(source_provider, timestamp, expected):
    provider, _ = source_provider
    payload = {"role": "user", "parts": [{"type": "text", "text": "Input"}]}
    if expected is not None:
        payload["created_at"] = expected
    assert provider._messages_to_openviking_batch([
        {"role": "user", "content": "Input", "timestamp": timestamp},
        {"role": "assistant", "content": "Reply", "tool_call_id": "not-a-message-id"},
    ]) == [payload, {"role": "assistant", "parts": [{"type": "text", "text": "Reply"}]}]


def call(call_id, name="shell"):
    return {"id": call_id, "function": {"name": name, "arguments": '{"command":"pwd"}'}}


@pytest.mark.parametrize("conflict", ["none", "different", "missing"])
def test_tool_aggregate_keeps_contributors_and_only_shared_scalars(source_provider, conflict):
    provider, _ = source_provider
    metadata = {"created_at": "2026-10-17T15:00:00Z", "turn_id": "turn-a",
                "message_kind": "tool_transport"}
    result_metadata = metadata if conflict == "none" else (
        {"created_at": "2026-10-17T15:00:01Z", "turn_id": "turn-b",
         "message_kind": "assistant_step"} if conflict == "different" else {}
    )
    batch = provider._messages_to_openviking_batch([
        {"role": "assistant", "tool_calls": [call("one"), call("two"), call("recall", "viking_read")],
         "message_uid": "call-owner", **metadata},
        {"role": "tool", "tool_call_id": "one", "content": "first", "message_uid": "result-one", **metadata},
        {"role": "tool", "tool_call_id": "recall", "content": "Recalled text", "message_uid": "excluded"},
        {"role": "tool", "tool_call_id": "two", "content": "second", "message_uid": "result-two", **result_metadata},
        {"role": "assistant", "content": "Done"},
    ])
    expected = {
        "role": "assistant", "parts": [{
            "type": "tool", "tool_id": "one", "tool_name": "shell", "tool_input": {"command": "pwd"},
            "tool_status": "completed", "tool_output": "first",
        }, {
            "type": "tool", "tool_id": "two", "tool_name": "shell", "tool_input": {"command": "pwd"},
            "tool_status": "completed", "tool_output": "second",
        }], "source_message_ids": ["call-owner", "result-one", "result-two"],
    }
    if conflict == "none":
        expected.update(metadata)
    assert batch == [expected, {"role": "assistant", "parts": [{"type": "text", "text": "Done"}]}]


def test_reused_tool_ids_do_not_borrow_unrelated_source_messages(source_provider):
    provider, _ = source_provider
    batch = provider._messages_to_openviking_batch([
        {"role": "assistant", "tool_calls": [call("same")], "message_uid": "first-call"},
        {"role": "tool", "tool_call_id": "same", "content": "one", "message_uid": "first-result"},
        {"role": "assistant", "tool_calls": [call("same")], "message_uid": "second-call"},
        {"role": "tool", "tool_call_id": "same", "content": "two", "message_uid": "second-result"},
        {"role": "user", "content": "Next"},
        {"role": "tool", "tool_call_id": "same", "content": "orphan", "message_uid": "orphan-result"},
    ])
    assert [message.get("source_message_ids") for message in batch] == [
        ["first-call", "first-result"], ["second-call", "second-result"], None, ["orphan-result"],
    ]
    assert [part.get("tool_output", part.get("text")) for m in batch for part in m["parts"]] == [
        "one", "two", "Next", "orphan",
    ]


@pytest.mark.parametrize("route", ["batch", "text", "serial", "retry"])
def test_public_sync_payloads_survive_mock_upload_routes(source_provider, monkeypatch, tmp_path, route):
    provider, module = source_provider
    requests = []
    accepted = []
    path = "/api/v1/sessions/source-session/messages"

    def handle(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body, request.headers["X-OpenViking-Actor-Peer"]))
        if request.url.path.endswith("/messages/batch"):
            fail = route == "serial" or (route == "text" and len(requests) == 1) or (
                route == "retry" and len(requests) == 2
            )
            if fail:
                provider._agent = "changed-after-capture"
                return httpx.Response(503, json={"error": {"message": "MOCK upload failure"}})
            accepted.extend(body["messages"])
        elif request.url.path == path:
            accepted.append(body)
        else:
            return httpx.Response(200, json={"result": {"pending_tokens": 0}})
        return httpx.Response(200, json={"result": {}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        monkeypatch.setattr(httpx, "post", transport.post)
        monkeypatch.setattr(httpx, "get", transport.get)
        monkeypatch.setattr(module, "_SESSION_MESSAGE_BATCH_LIMIT", 2)
        provider._agent = "assistant-peer"
        provider._client = module._VikingClient(
            "http://openviking.test", account="test", user="test", agent="assistant-peer"
        )
        provider._session_id = "source-session"
        source = [
            {"role": "user", "content": "Old", "message_uid": "excluded-history"},
            {"role": "assistant", "content": "Old reply"},
            {"role": "user", "content": "Input", "timestamp": 0, "platform_message_id": "platform-7"},
            {"role": "assistant", "tool_calls": [call("one")], "turn_id": "turn-a",
             "message_kind": "tool_transport", "message_uid": "call-owner"},
            {"role": "tool", "tool_call_id": "one", "content": "result", "turn_id": "turn-a",
             "message_kind": "tool_transport", "message_uid": "tool-result"},
            {"role": "assistant", "content": "Reply", "message_uid": "final-reply"},
        ]
        original = copy.deepcopy(source)
        provider.sync_turn("Input", "Reply", session_id="source-session", messages=source,
                           turn_author={"id": "sender", "name": "Unverified author", "timestamp": "not-source"})
        assert provider._drain_writers("source-session", timeout=5)
        assert source == original

    expected = [
        {"role": "user", "parts": [{"type": "text", "text": "Input"}],
         "created_at": "1970-01-01T00:00:00+00:00", "source_message_ids": ["platform-7"]},
        {"role": "assistant", "parts": [{"type": "tool", "tool_id": "one", "tool_name": "shell",
         "tool_input": {"command": "pwd"}, "tool_status": "completed", "tool_output": "result"}],
         "peer_id": "assistant-peer", "turn_id": "turn-a", "message_kind": "tool_transport",
         "source_message_ids": ["call-owner", "tool-result"]},
        {"role": "assistant", "parts": [{"type": "text", "text": "Reply"}],
         "peer_id": "assistant-peer", "source_message_ids": ["final-reply"]},
    ]
    if route == "text":
        expected = [
            {"role": "user", "parts": [{"type": "text", "text": "Input"}]},
            {"role": "assistant", "parts": [{"type": "text", "text": "Reply"}], "peer_id": "assistant-peer"},
        ]
    assert accepted == expected
    assert {peer for _, _, peer in requests} == {"assistant-peer"}
    if route == "serial":
        assert [body for url, body, _ in requests if url == path] == expected
        assert requests[0][1] == requests[2][1]
    if route == "retry":
        assert requests[1] == requests[2]
    (tmp_path / "accepted.json").write_text(json.dumps(accepted), encoding="utf-8")
    assert json.loads((tmp_path / "accepted.json").read_text()) == expected


def test_unknown_source_fields_stay_absent_beside_known_metadata(source_provider):
    provider, _ = source_provider
    assert provider._messages_to_openviking_batch([
        {"role": "user", "content": "Input", "source_message_ids": [None, 42, ""],
         "turn_id": 42, "message_kind": None, "message_uid": None, "id": "untyped-id",
         "platform_message_id": False, "timestamp": None},
        {"role": "assistant", "content": "Reply", "turn_id": "native-turn"},
    ]) == [
        {"role": "user", "parts": [{"type": "text", "text": "Input"}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "Reply"}], "turn_id": "native-turn"},
    ]


def test_native_redaction_retains_supplied_fields_without_mutating_input(source_provider):
    from agent.memory_manager import _redact_for_provider

    provider, _ = source_provider
    message = {"role": "user", "content": "Input", "timestamp": 1.25,
               "platform_message_id": "platform-42", "message_uid": "durable-7"}
    original = copy.deepcopy(message)
    clean = _redact_for_provider([message])
    assert provider._messages_to_openviking_batch(clean) == [{
        "role": "user", "parts": [{"type": "text", "text": "Input"}],
        "created_at": "1970-01-01T00:00:01.250000+00:00",
        "source_message_ids": ["durable-7", "platform-42"],
    }]
    assert message == original
