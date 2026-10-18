"""Origin wire boundary for the standalone HTTP provider (no SDK dependency)."""

import json
import math
import re
from datetime import datetime


class OriginContractMismatch(ValueError):
    """An append may have landed; no retry, fallback, or cursor advance is safe."""


def _object(value, required, optional=()):
    if (
        not isinstance(value, dict)
        or set(required) - value.keys()
        or value.keys() - set(required) - set(optional)
    ):
        raise OriginContractMismatch("Incomplete or unsupported origin object")
    return value


def _text(value):
    if not isinstance(value, str) or not value:
        raise OriginContractMismatch("Expected a nonempty origin string")
    return value


def _optional_text(value):
    if value is not None and not isinstance(value, str):
        raise OriginContractMismatch("Invalid optional origin string")
    return value


def _rfc3339(value):
    _text(value)
    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:(?:[0-5]\d|60)(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})", value
    ):
        raise OriginContractMismatch("Invalid origin timestamp")
    check = value[:17] + ("59" if value[17:19] == "60" else value[17:19]) + value[19:]
    try:
        datetime.fromisoformat(check.replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError as exc:
        raise OriginContractMismatch("Invalid origin timestamp") from exc
    return value


def observed_time(literal, meaning, basis):
    encoding = "unix_seconds" if type(literal) in (int, float) else "opaque"
    if isinstance(literal, str):
        try:
            _rfc3339(literal)
            encoding = "rfc3339"
        except OriginContractMismatch:
            pass
    return {"literal": literal, "encoding": encoding, "meaning": meaning, "basis": basis}


def _basis(value):
    value = _object(value, ("producer", "field"))
    return {"producer": _text(value["producer"]), "field": _text(value["field"])}


def _actor(value):
    if value is None:
        return None
    value = _object(value, ("actor", "basis"))
    actor = _object(value["actor"], ("identifier",), ("scope",))
    return {
        "actor": {
            "identifier": _text(actor["identifier"]),
            "scope": _optional_text(actor.get("scope")),
        },
        "basis": _basis(value["basis"]),
    }


def statements(value):
    value = _object(value, ("version", "contributors"))
    if (
        type(value["version"]) is not int
        or value["version"] != 1
        or not isinstance(value["contributors"], list)
    ):
        raise OriginContractMismatch("Unsupported origin statement schema")
    contributors = []
    for item in value["contributors"]:
        item = _object(
            item,
            ("locator",),
            ("refs", "times", "transport_sender", "primary_author", "content_class", "class_basis"),
        )
        locator = _object(item["locator"], ("provider",), ("session_id", "source_scope"))
        if not isinstance(item.get("refs", []), list) or not isinstance(
            item.get("times", []), list
        ):
            raise OriginContractMismatch("Origin refs/times must be arrays")
        refs, times = [], []
        for ref in item.get("refs", []):
            ref = _object(ref, ("kind", "value"))
            if ref["kind"] not in ("provider_record", "host_row", "source_message", "supplied_ref"):
                raise OriginContractMismatch("Unsupported origin reference")
            refs.append({"kind": ref["kind"], "value": _text(ref["value"])})
        for time in item.get("times", []):
            time = _object(time, ("literal", "encoding", "meaning", "basis"))
            literal = time["literal"]
            numeric = type(literal) in (int, float) and (
                not isinstance(literal, float) or math.isfinite(literal)
            )
            if time["encoding"] == "rfc3339":
                _rfc3339(literal)
            elif not (
                time["encoding"] == "unix_seconds"
                and numeric
                or time["encoding"] == "opaque"
                and (numeric or isinstance(literal, str))
            ):
                raise OriginContractMismatch("Invalid origin time literal")
            if time["meaning"] not in ("source_event", "provider_record", "host_stamp", "unknown"):
                raise OriginContractMismatch("Unsupported origin time meaning")
            times.append({**time, "basis": _basis(time["basis"])})
        content_class = item.get("content_class", "unknown")
        if (
            content_class
            not in ("unknown", "direct_input", "forwarded", "runtime_self_metadata", "bb_relay")
            or content_class != "unknown"
            and item.get("class_basis") is None
        ):
            raise OriginContractMismatch("Unsupported origin class or missing basis")
        contributors.append(
            {
                "locator": {
                    "provider": _text(locator["provider"]),
                    "session_id": _optional_text(locator.get("session_id")),
                    "source_scope": _optional_text(locator.get("source_scope")),
                },
                "refs": refs,
                "times": times,
                "transport_sender": _actor(item.get("transport_sender")),
                "primary_author": _actor(item.get("primary_author")),
                "content_class": content_class,
                "class_basis": _basis(item["class_basis"])
                if item.get("class_basis") is not None
                else None,
            }
        )
    return {"version": 1, "contributors": contributors}


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def validate_readback(messages, response):
    if not any(message.get("origin") is not None for message in messages):
        return
    result = response.get("result") if isinstance(response, dict) else None
    if (
        not isinstance(result, dict)
        or result.get("origin_readback") != "jsonl"
        or not isinstance(result.get("accepted_origins"), list)
    ):
        raise OriginContractMismatch("Server omitted persistent origin readback")
    receipts = result["accepted_origins"]
    ids, groups, clocks = set(), set(), set()
    for index, message in enumerate(messages):
        parts = message.get("parts") or []
        count = (
            len(parts)
            if message.get("role") == "user"
            and len(parts) > 1
            and all(part.get("type") == "tool" for part in parts)
            else 1
        )
        members = [
            item
            for item in receipts
            if isinstance(item, dict)
            and type(item.get("input_index")) is int
            and item["input_index"] == index
        ]
        if len(members) != count:
            raise OriginContractMismatch("Incomplete origin input/fanout coverage")
        expected = _canonical(
            statements(message.get("origin") or {"version": 1, "contributors": []})
        )
        member_groups, indices = set(), set()
        for member in members:
            member = _object(
                member,
                (
                    "input_index",
                    "message_id",
                    "origin",
                    "legacy_created_at",
                    "turn_id",
                    "message_kind",
                    "source_message_ids",
                ),
            )
            message_id = _text(member["message_id"])
            if message_id in ids:
                raise OriginContractMismatch("Duplicate origin message ID")
            ids.add(message_id)
            origin = _object(member["origin"], ("version", "contributors", "receiver"))
            if (
                _canonical(
                    statements({key: value for key, value in origin.items() if key != "receiver"})
                )
                != expected
            ):
                raise OriginContractMismatch("Stored contributor literals changed")
            receiver = _object(origin["receiver"], ("session_ingest_started_at", "applicability"))
            stamp = _rfc3339(receiver["session_ingest_started_at"])
            clocks.add(stamp)
            scope = _object(
                receiver["applicability"], ("kind", "group_id"), ("member_count", "member_index")
            )
            member_groups.add(_text(scope["group_id"]))
            if count == 1:
                if (
                    scope["kind"] != "input_message"
                    or "member_count" in scope
                    or "member_index" in scope
                ):
                    raise OriginContractMismatch("Invalid whole-input origin scope")
            else:
                slot = scope.get("member_index")
                if (
                    scope["kind"] != "input_group"
                    or type(scope.get("member_count")) is not int
                    or scope["member_count"] != count
                    or type(slot) is not int
                    or not 0 <= slot < count
                    or slot in indices
                ):
                    raise OriginContractMismatch("Invalid fanout origin scope")
                indices.add(slot)
            if (
                member["legacy_created_at"] != (message.get("created_at") or stamp)
                or member["turn_id"] != message.get("turn_id")
                or member["message_kind"]
                != (message.get("message_kind") or ("tool_transport" if count > 1 else None))
                or member["source_message_ids"] != message.get("source_message_ids")
            ):
                raise OriginContractMismatch("Stored source fields changed")
        if len(member_groups) != 1 or member_groups & groups:
            raise OriginContractMismatch("Origin group reused across inputs")
        groups.update(member_groups)
    if len(ids) != len(receipts) or len(clocks) != 1:
        raise OriginContractMismatch("Unexpected origin members or clocks")
