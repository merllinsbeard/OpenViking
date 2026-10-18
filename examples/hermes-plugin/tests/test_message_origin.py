"""Native upload/metadata functions with explicitly MOCK HTTP transport."""

import copy
import json
from pathlib import Path

import httpx
import pytest


def fixture():
    return json.loads(
        (Path(__file__).resolve().parents[3] / "tests/fixtures/message-origin-v1.json").read_text()
    )


@pytest.mark.parametrize(
    "fault", ["omitted", "partial", "literal", "version", "duplicate", "strict", "lost"]
)
def test_real_turn_upload_holds_next_index_without_fallback(source_provider, monkeypatch, fault):
    _, module = source_provider
    golden = fixture()
    response = golden["response"]
    member = response["result"]["accepted_origins"][0]
    if fault == "omitted":
        response["result"].pop("accepted_origins")
    elif fault == "partial":
        member["origin"].pop("receiver")
    elif fault == "literal":
        member["origin"]["contributors"][0]["times"][0]["literal"] = "2026-10-17T02:10:11.123456Z"
    elif fault == "version":
        member["origin"]["version"] = 2
    elif fault == "duplicate":
        response["result"]["accepted_origins"].append(copy.deepcopy(member))
    calls = []

    def handle(request):
        calls.append(request)
        if fault == "lost":
            raise httpx.ReadTimeout("MOCK ACK lost after append", request=request)
        return httpx.Response(422 if fault == "strict" else 200, json=response)

    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        monkeypatch.setattr(httpx, "post", transport.post)
        client = module._VikingClient("http://openviking.test", account="test", user="test")
        upload = module._TurnUpload(
            client, "native-v1", [golden["message"]], "Text fallback must not run", "Reply", ""
        )
        assert upload.run() is None
    assert upload.next_index == 0
    assert len(calls) == 1
    assert calls[0].url.path.endswith("/messages/batch")


def test_native_turn_upload_advances_only_the_verified_prefix(
    source_provider, origin_ack, monkeypatch
):
    _, module = source_provider
    first = fixture()["message"]
    tail = {**copy.deepcopy(first), "content": "Uncertain tail"}
    calls = []

    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        response = origin_ack(body["messages"])
        if len(calls) == 2:
            response["result"]["accepted_origins"][0]["origin"]["contributors"] = []
        return httpx.Response(200, json=response)

    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        monkeypatch.setattr(httpx, "post", transport.post)
        monkeypatch.setattr(module, "_SESSION_MESSAGE_BATCH_LIMIT", 1)
        client = module._VikingClient("http://openviking.test", account="test", user="test")
        upload = module._TurnUpload(
            client, "native-v1", [first, tail], "Must not fallback", "Reply", ""
        )
        assert upload.run() is None
    assert upload.next_index == 1
    assert len(calls) == 2


@pytest.mark.parametrize("meaning", ["host_stamp", "source_event", "unknown"])
def test_numeric_zero_keeps_native_marker_or_unknown_basis(source_provider, meaning):
    provider, _ = source_provider
    message = {"role": "user", "content": "Forwarded words", "timestamp": 0, "message_uid": "row-A"}
    if meaning != "unknown":
        message["origin_statements"] = {
            "version": 1,
            "contributors": [
                {
                    "locator": {"provider": "hermes", "session_id": "native-session"},
                    "refs": [{"kind": "host_row", "value": "row-A"}],
                    "times": [
                        {
                            "literal": 0,
                            "encoding": "unix_seconds",
                            "meaning": meaning,
                            "basis": {
                                "producer": "hermes",
                                "field": "stamp_message_timestamp"
                                if meaning == "host_stamp"
                                else "platform.event_time",
                            },
                        }
                    ],
                }
            ],
        }
    payload = provider._messages_to_openviking_batch([message])[0]
    time = payload["origin"]["contributors"][0]["times"][0]
    assert time["literal"] == 0 and type(time["literal"]) is int
    assert time["meaning"] == meaning
    assert payload["created_at"] == "1970-01-01T00:00:00+00:00"
    assert payload["origin"]["contributors"][0]["primary_author"] is None


def test_absorbed_uid_without_facts_does_not_borrow_survivor_time(source_provider):
    provider, _ = source_provider
    payload = provider._messages_to_openviking_batch(
        [
            {
                "role": "user",
                "content": "Merged input",
                "timestamp": 0,
                "message_uid": "survivor",
                "_absorbed_message_uids": ["absorbed"],
            }
        ]
    )[0]
    contributors = payload["origin"]["contributors"]
    assert len(contributors) == 2
    assert contributors[0]["refs"] == [{"kind": "host_row", "value": "survivor"}]
    assert contributors[0]["times"][0]["literal"] == 0
    assert contributors[1]["refs"] == [{"kind": "host_row", "value": "absorbed"}]
    assert contributors[1]["times"] == []


def test_marked_runtime_and_relay_are_claims_and_never_role_inferences(source_provider):
    provider, _ = source_provider
    messages = [{"role": "user", "content": "Unmarked"}]
    for marker in ("runtime_self_metadata", "bb_relay"):
        messages.append(
            {
                "role": "user",
                "content": "Native marked input",
                "origin_statements": {
                    "version": 1,
                    "contributors": [
                        {
                            "locator": {"provider": "hermes"},
                            "content_class": marker,
                            "class_basis": {"producer": "hermes", "field": "native.marker"},
                        }
                    ],
                },
            }
        )
    payloads = provider._messages_to_openviking_batch(messages)
    assert [payload["origin"]["contributors"][0]["content_class"] for payload in payloads] == [
        "unknown",
        "runtime_self_metadata",
        "bb_relay",
    ]
    assert all(
        payload["origin"]["contributors"][0]["primary_author"] is None for payload in payloads
    )
