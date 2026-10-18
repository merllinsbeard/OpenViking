# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import json

import pytest

from openviking.message import Message
from openviking.message.origin import _parse_origin_input


def source_statements():
    return {
        "version": 1,
        "contributors": [
            {
                "locator": {
                    "provider": "synthetic",
                    "session_id": "session-A",
                    "source_scope": None,
                },
                "refs": [{"kind": "provider_record", "value": "record-A"}],
                "times": [
                    {
                        "literal": "2026-10-17T09:10:11.123456+07:00",
                        "encoding": "rfc3339",
                        "meaning": "source_event",
                        "basis": {"producer": "synthetic", "field": "event.time"},
                    }
                ],
                "transport_sender": {
                    "actor": {"identifier": "sender-A", "scope": "chat-A"},
                    "basis": {"producer": "synthetic", "field": "event.sender"},
                },
                "primary_author": None,
                "content_class": "unknown",
                "class_basis": None,
            }
        ],
    }


def recorded_origin():
    return {
        **source_statements(),
        "receiver": {
            "session_ingest_started_at": "2026-10-17T00:00:00Z",
            "applicability": {"kind": "input_message", "group_id": "group-A"},
        },
    }


def test_recorded_origin_survives_jsonl_without_inventing_authorship():
    expected = {
        "id": "msg-A",
        "role": "user",
        "parts": [{"type": "text", "text": "转发文字"}],
        "created_at": "2026-10-17T09:10:11.123456+07:00",
        "turn_id": "turn-A",
        "source_message_ids": ["record-A"],
        "origin": recorded_origin(),
    }
    row = expected
    for _ in range(3):
        row = json.loads(Message.from_dict(row).to_jsonl())
        assert row == expected


@pytest.mark.parametrize(
    "literal, encoding",
    [(0, "unix_seconds"), (0.25, "unix_seconds"), ("0", "opaque"), (0, "opaque")],
)
def test_time_json_scalar_and_meaning_remain_literal(literal, encoding):
    origin = recorded_origin()
    origin["contributors"][0]["times"][0].update(
        literal=literal, encoding=encoding, meaning="host_stamp"
    )
    message = Message.from_dict({"id": "epoch", "role": "user", "parts": [], "origin": origin})
    time = json.loads(message.to_jsonl())["origin"]["contributors"][0]["times"][0]
    assert time == {
        "literal": literal,
        "encoding": encoding,
        "meaning": "host_stamp",
        "basis": {"producer": "synthetic", "field": "event.time"},
    }
    assert type(time["literal"]) is type(literal)


@pytest.mark.parametrize(
    "origin",
    [
        {"version": 2, "contributors": []},
        {"version": True, "contributors": []},
        {"version": 1, "contributors": [], "receiver": {}},
        {"version": 1, "contributors": [], "session_ingest_started_at": "1900-01-01T00:00:00Z"},
    ],
)
def test_inbound_origin_rejects_receiver_or_unknown_schema(origin):
    with pytest.raises(ValueError):
        _parse_origin_input(origin)


@pytest.mark.parametrize(
    "literal, encoding",
    [
        (True, "unix_seconds"),
        (float("inf"), "unix_seconds"),
        ("0", "unix_seconds"),
        ("2026-10-17", "rfc3339"),
        ("2026-10-17T09:10:11", "rfc3339"),
    ],
)
def test_inbound_time_requires_declared_encoding(literal, encoding):
    origin = source_statements()
    origin["contributors"][0]["times"][0].update(literal=literal, encoding=encoding)
    with pytest.raises(ValueError):
        _parse_origin_input(origin)


def test_known_content_class_requires_native_basis():
    origin = source_statements()
    origin["contributors"][0]["content_class"] = "direct_input"
    with pytest.raises(ValueError):
        _parse_origin_input(origin)


def test_legacy_decode_keeps_unknown_receiver_and_missing_date():
    row = {"id": "old", "role": "user", "content": "legacy"}
    for _ in range(2):
        row = json.loads(Message.from_dict(row).to_jsonl())
        assert row == {
            "id": "old",
            "role": "user",
            "parts": [{"type": "text", "text": "legacy"}],
            "created_at": None,
        }
