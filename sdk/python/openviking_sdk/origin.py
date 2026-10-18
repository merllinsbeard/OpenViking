# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Message contributor claims and checked native append receipts."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Literal, Optional, Tuple, Union

from .errors import OpenVikingError


@dataclass(frozen=True)
class FieldBasis:
    producer: str
    field: str


@dataclass(frozen=True)
class SourceLocator:
    provider: str
    session_id: Optional[str] = None
    source_scope: Optional[str] = None


@dataclass(frozen=True)
class SourceRef:
    kind: Literal["provider_record", "host_row", "source_message", "supplied_ref"]
    value: str


@dataclass(frozen=True)
class TimeStatement:
    literal: Union[str, int, float]
    encoding: Literal["rfc3339", "unix_seconds", "opaque"]
    meaning: Literal["source_event", "provider_record", "host_stamp", "unknown"]
    basis: FieldBasis


@dataclass(frozen=True)
class ActorRef:
    identifier: str
    scope: Optional[str] = None


@dataclass(frozen=True)
class ActorClaim:
    actor: ActorRef
    basis: FieldBasis


@dataclass(frozen=True)
class ContributorStatement:
    locator: SourceLocator
    refs: Tuple[SourceRef, ...] = ()
    times: Tuple[TimeStatement, ...] = ()
    transport_sender: Optional[ActorClaim] = None
    primary_author: Optional[ActorClaim] = None
    content_class: Literal[
        "direct_input", "forwarded", "runtime_self_metadata", "bb_relay", "unknown"
    ] = "unknown"
    class_basis: Optional[FieldBasis] = None


@dataclass(frozen=True)
class OriginStatements:
    contributors: Tuple[ContributorStatement, ...] = ()
    version: Literal[1] = 1


@dataclass(frozen=True)
class WholeInputMessage:
    group_id: str
    kind: Literal["input_message"] = "input_message"


@dataclass(frozen=True)
class FanoutGroup:
    group_id: str
    member_index: int
    member_count: int
    kind: Literal["input_group"] = "input_group"


@dataclass(frozen=True)
class IngestStamp:
    session_ingest_started_at: str
    applicability: Union[WholeInputMessage, FanoutGroup]


@dataclass(frozen=True)
class RecordedOrigin:
    statements: OriginStatements
    receiver: IngestStamp


@dataclass(frozen=True)
class AcceptedOrigin:
    input_index: int
    message_id: str
    origin: RecordedOrigin
    legacy_created_at: Optional[str]
    turn_id: Optional[str]
    message_kind: Optional[str]
    source_message_ids: Optional[Tuple[str, ...]]


class OriginContractMismatch(OpenVikingError):
    """The append may have completed; do not retry or advance a capture cursor."""


def _object(value, required, optional=()):
    if (
        not isinstance(value, dict)
        or set(required) - value.keys()
        or value.keys() - set(required) - set(optional)
    ):
        raise OriginContractMismatch("Malformed origin contract object")
    return value


def _string(value, *, optional=False):
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value:
        raise OriginContractMismatch("Expected a nonempty origin string")
    return value


def _optional_text(value):
    if value is not None and not isinstance(value, str):
        raise OriginContractMismatch("Expected an optional origin string")
    return value


def _rfc3339(value):
    _string(value)
    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:(?:[0-5]\d|60)(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})", value
    ):
        raise OriginContractMismatch("Expected an origin timestamp with an explicit timezone")
    check = value[:17] + ("59" if value[17:19] == "60" else value[17:19]) + value[19:]
    try:
        datetime.fromisoformat(check.replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError as exc:
        raise OriginContractMismatch("Invalid origin timestamp") from exc
    return value


def _basis(value):
    value = _object(value, ("producer", "field"))
    return FieldBasis(_string(value["producer"]), _string(value["field"]))


def _actor(value):
    if value is None:
        return None
    value = _object(value, ("actor", "basis"))
    actor = _object(value["actor"], ("identifier",), ("scope",))
    return ActorClaim(
        actor=ActorRef(_string(actor["identifier"]), _optional_text(actor.get("scope"))),
        basis=_basis(value["basis"]),
    )


def _decode_statements(value) -> OriginStatements:
    value = _object(value, ("version", "contributors"))
    if (
        type(value["version"]) is not int
        or value["version"] != 1
        or not isinstance(value["contributors"], (list, tuple))
    ):
        raise OriginContractMismatch("Unsupported origin statement schema")
    contributors = []
    for contributor in value["contributors"]:
        contributor = _object(
            contributor,
            ("locator",),
            ("refs", "times", "transport_sender", "primary_author", "content_class", "class_basis"),
        )
        locator = _object(contributor["locator"], ("provider",), ("session_id", "source_scope"))
        refs = []
        times = []
        if not isinstance(contributor.get("refs", ()), (list, tuple)) or not isinstance(
            contributor.get("times", ()), (list, tuple)
        ):
            raise OriginContractMismatch("Origin refs and times must be collections")
        for ref in contributor.get("refs", ()):
            ref = _object(ref, ("kind", "value"))
            if ref["kind"] not in ("provider_record", "host_row", "source_message", "supplied_ref"):
                raise OriginContractMismatch("Unsupported source reference kind")
            refs.append(SourceRef(ref["kind"], _string(ref["value"])))
        for statement in contributor.get("times", ()):
            statement = _object(statement, ("literal", "encoding", "meaning", "basis"))
            literal = statement["literal"]
            encoding = statement["encoding"]
            if encoding == "unix_seconds":
                if type(literal) not in (int, float) or (
                    isinstance(literal, float) and not math.isfinite(literal)
                ):
                    raise OriginContractMismatch("Invalid numeric time literal")
            elif encoding == "rfc3339":
                _rfc3339(literal)
            elif (
                encoding != "opaque"
                or type(literal) not in (str, int, float)
                or (isinstance(literal, float) and not math.isfinite(literal))
            ):
                raise OriginContractMismatch("Unsupported time encoding")
            if statement["meaning"] not in (
                "source_event",
                "provider_record",
                "host_stamp",
                "unknown",
            ):
                raise OriginContractMismatch("Unsupported time meaning")
            times.append(
                TimeStatement(literal, encoding, statement["meaning"], _basis(statement["basis"]))
            )
        content_class = contributor.get("content_class", "unknown")
        if content_class not in (
            "direct_input",
            "forwarded",
            "runtime_self_metadata",
            "bb_relay",
            "unknown",
        ):
            raise OriginContractMismatch("Unsupported content class")
        class_basis = contributor.get("class_basis")
        if content_class != "unknown" and class_basis is None:
            raise OriginContractMismatch("Known content class requires a native basis")
        contributors.append(
            ContributorStatement(
                locator=SourceLocator(
                    _string(locator["provider"]),
                    _optional_text(locator.get("session_id")),
                    _optional_text(locator.get("source_scope")),
                ),
                refs=tuple(refs),
                times=tuple(times),
                transport_sender=_actor(contributor.get("transport_sender")),
                primary_author=_actor(contributor.get("primary_author")),
                content_class=content_class,
                class_basis=_basis(class_basis) if class_basis is not None else None,
            )
        )
    return OriginStatements(contributors=tuple(contributors))


def _encode_origin(value: OriginStatements) -> Dict[str, Any]:
    if not isinstance(value, OriginStatements):
        raise TypeError("origin must be an OriginStatements value")
    statements = _decode_statements(asdict(value))
    return json.loads(json.dumps(asdict(statements), ensure_ascii=False, allow_nan=False))


def _validate_origin_readback(messages, result):
    if not any(message.get("origin") is not None for message in messages):
        return result
    if (
        not isinstance(result, dict)
        or result.get("origin_readback") != "jsonl"
        or not isinstance(result.get("accepted_origins"), list)
    ):
        raise OriginContractMismatch("Server did not return a persistent origin readback")
    receipts = result["accepted_origins"]
    seen_ids = set()
    seen_groups: set[str] = set()
    accepted = []
    batch_stamps = set()
    for input_index, message in enumerate(messages):
        members = [
            item
            for item in receipts
            if isinstance(item, dict)
            and type(item.get("input_index")) is int
            and item["input_index"] == input_index
        ]
        parts = message.get("parts") or []
        count = (
            len(parts)
            if message.get("role") == "user"
            and len(parts) > 1
            and all(part.get("type") == "tool" for part in parts)
            else 1
        )
        if len(members) != count:
            raise OriginContractMismatch("Origin readback has incomplete input/fanout coverage")
        expected_statements = _decode_statements(
            message.get("origin") or {"version": 1, "contributors": []}
        )
        member_indices = set()
        group_ids = set()
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
            message_id = _string(member["message_id"])
            if message_id in seen_ids:
                raise OriginContractMismatch("Duplicate origin message ID")
            seen_ids.add(message_id)
            origin = _object(member["origin"], ("version", "contributors", "receiver"))
            statements = _decode_statements(
                {key: value for key, value in origin.items() if key != "receiver"}
            )
            if json.dumps(asdict(statements), sort_keys=True) != json.dumps(
                asdict(expected_statements), sort_keys=True
            ):
                raise OriginContractMismatch("Origin contributor literals changed in storage")
            receiver = _object(origin["receiver"], ("session_ingest_started_at", "applicability"))
            stamp = _rfc3339(receiver["session_ingest_started_at"])
            batch_stamps.add(stamp)
            scope = _object(
                receiver["applicability"], ("kind", "group_id"), ("member_index", "member_count")
            )
            group_id = _string(scope["group_id"])
            group_ids.add(group_id)
            applicability: Union[WholeInputMessage, FanoutGroup]
            if count == 1:
                if (
                    scope["kind"] != "input_message"
                    or "member_index" in scope
                    or "member_count" in scope
                ):
                    raise OriginContractMismatch("Invalid whole-input scope")
                applicability = WholeInputMessage(group_id)
            else:
                index = scope.get("member_index")
                if (
                    scope["kind"] != "input_group"
                    or type(index) is not int
                    or not 0 <= index < count
                    or type(scope.get("member_count")) is not int
                    or scope["member_count"] != count
                    or index in member_indices
                ):
                    raise OriginContractMismatch("Invalid fanout member scope")
                member_indices.add(index)
                applicability = FanoutGroup(group_id, index, count)
            expected_kind = message.get("message_kind") or ("tool_transport" if count > 1 else None)
            expected_date = message.get("created_at") or stamp
            if (
                member["legacy_created_at"] != expected_date
                or member["turn_id"] != message.get("turn_id")
                or member["message_kind"] != expected_kind
                or member["source_message_ids"] != message.get("source_message_ids")
            ):
                raise OriginContractMismatch("Origin readback source fields changed")
            accepted.append(
                AcceptedOrigin(
                    input_index,
                    message_id,
                    RecordedOrigin(statements, IngestStamp(stamp, applicability)),
                    member["legacy_created_at"],
                    member["turn_id"],
                    member["message_kind"],
                    tuple(member["source_message_ids"])
                    if member["source_message_ids"] is not None
                    else None,
                )
            )
        if len(group_ids) != 1 or group_ids & seen_groups:
            raise OriginContractMismatch("Origin group identity is incomplete or reused")
        seen_groups.update(group_ids)
    if len(accepted) != len(receipts) or len(batch_stamps) != 1:
        raise OriginContractMismatch("Origin readback contains unexpected members or ingest clocks")
    return {**result, "accepted_origins": accepted}
