# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Contributor claims, receiver facts, and historical extraction selections."""

import re
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import (
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    TypeAdapter,
    field_validator,
    model_validator,
)
from pydantic.dataclasses import dataclass

if TYPE_CHECKING:
    from openviking.message.message import Message

_CONFIG = ConfigDict(extra="forbid", allow_inf_nan=False)
_Nonempty = Annotated[StrictStr, Field(min_length=1)]
_Index = Annotated[StrictInt, Field(ge=0)]
_Count = Annotated[StrictInt, Field(ge=1)]


def _validate_rfc3339(value: str) -> str:
    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:(?:[0-5]\d|60)(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})",
        value,
    ):
        raise ValueError("Expected an RFC3339 literal with an explicit timezone")
    check = value[:17] + ("59" if value[17:19] == "60" else value[17:19]) + value[19:]
    datetime.fromisoformat(check.replace("z", "+00:00").replace("Z", "+00:00"))
    return value


@dataclass(frozen=True, config=_CONFIG)
class FieldBasis:
    producer: _Nonempty
    field: _Nonempty


@dataclass(frozen=True, config=_CONFIG)
class SourceLocator:
    provider: _Nonempty
    session_id: StrictStr | None = None
    source_scope: StrictStr | None = None


@dataclass(frozen=True, config=_CONFIG)
class SourceRef:
    kind: Literal["provider_record", "host_row", "source_message", "supplied_ref"]
    value: _Nonempty


@dataclass(frozen=True, config=_CONFIG)
class TimeStatement:
    literal: StrictStr | StrictInt | StrictFloat
    encoding: Literal["rfc3339", "unix_seconds", "opaque"]
    meaning: Literal["source_event", "provider_record", "host_stamp", "unknown"]
    basis: FieldBasis

    @model_validator(mode="after")
    def validate_literal(self) -> "TimeStatement":
        if self.encoding == "unix_seconds":
            if not isinstance(self.literal, (int, float)):
                raise ValueError("unix_seconds requires a numeric JSON literal")
        elif self.encoding == "rfc3339":
            if not isinstance(self.literal, str):
                raise ValueError("RFC3339 requires a string literal")
            _validate_rfc3339(self.literal)
        return self


@dataclass(frozen=True, config=_CONFIG)
class ActorRef:
    identifier: _Nonempty
    scope: StrictStr | None = None


@dataclass(frozen=True, config=_CONFIG)
class ActorClaim:
    actor: ActorRef
    basis: FieldBasis


@dataclass(frozen=True, config=_CONFIG)
class ContributorStatement:
    locator: SourceLocator
    refs: tuple[SourceRef, ...] = ()
    times: tuple[TimeStatement, ...] = ()
    transport_sender: ActorClaim | None = None
    primary_author: ActorClaim | None = None
    content_class: Literal[
        "direct_input", "forwarded", "runtime_self_metadata", "bb_relay", "unknown"
    ] = "unknown"
    class_basis: FieldBasis | None = None

    @model_validator(mode="after")
    def validate_class_basis(self) -> "ContributorStatement":
        if self.content_class != "unknown" and self.class_basis is None:
            raise ValueError("A known content class requires its native field basis")
        return self


@dataclass(frozen=True, config=_CONFIG)
class OriginStatements:
    contributors: tuple[ContributorStatement, ...] = ()
    version: Literal[1] = 1

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("Unsupported origin version")
        return value


@dataclass(frozen=True, config=_CONFIG)
class WholeInputMessage:
    group_id: _Nonempty
    kind: Literal["input_message"] = "input_message"


@dataclass(frozen=True, config=_CONFIG)
class FanoutGroup:
    group_id: _Nonempty
    member_index: _Index
    member_count: Annotated[StrictInt, Field(ge=2)]
    kind: Literal["input_group"] = "input_group"

    @model_validator(mode="after")
    def validate_member(self) -> "FanoutGroup":
        if self.member_index >= self.member_count:
            raise ValueError("Fanout member index must be below member count")
        return self


@dataclass(frozen=True, config=_CONFIG)
class IngestStamp:
    """OV processing start, after batch validation and before build or append."""

    session_ingest_started_at: StrictStr
    applicability: Annotated[WholeInputMessage | FanoutGroup, Field(discriminator="kind")]

    @field_validator("session_ingest_started_at")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        return _validate_rfc3339(value)


@dataclass(frozen=True, config=_CONFIG)
class RecordedOrigin:
    statements: OriginStatements
    receiver: IngestStamp


@dataclass(frozen=True, config=_CONFIG)
class MessageFacts:
    message_id: _Nonempty
    origin: RecordedOrigin | None
    legacy_created_at: StrictStr | None
    turn_id: StrictStr | None
    message_kind: StrictStr | None
    source_message_ids: tuple[StrictStr, ...] | None

    @property
    def origin_statements(self) -> OriginStatements:
        return self.origin.statements if self.origin is not None else OriginStatements()


@dataclass(frozen=True, config=_CONFIG)
class ChunkSelection:
    index: _Index
    count: _Count

    @model_validator(mode="after")
    def validate_index(self) -> "ChunkSelection":
        if self.index >= self.count:
            raise ValueError("Chunk index must be below chunk count")
        return self


@dataclass(frozen=True, config=_CONFIG)
class SelectedMessage:
    facts: MessageFacts
    chunks: tuple[ChunkSelection, ...] = ()


@dataclass(frozen=True, config=_CONFIG)
class OriginSelection:
    mode: Literal["selected_ranges", "extraction_batch_candidates"]
    messages: tuple[SelectedMessage, ...]
    view_ranges: tuple[tuple[_Index, _Index], ...] = ()


@dataclass(frozen=True, config=_CONFIG)
class OriginConflict:
    extraction_id: str
    message_id: str | None
    field: str
    variants: tuple[object, ...]


class OriginContractMismatch(ValueError):
    """An append may have completed, but its origin receipt is unconfirmed."""


def _parse_origin_input(value: object) -> OriginStatements:
    if isinstance(value, OriginStatements):
        value = asdict(value)
    return TypeAdapter(OriginStatements).validate_python(value)


def _origin_to_dict(origin: RecordedOrigin) -> dict:
    return {
        **TypeAdapter(OriginStatements).dump_python(origin.statements, mode="json"),
        "receiver": TypeAdapter(IngestStamp).dump_python(origin.receiver, mode="json"),
    }


def _decode_recorded_origin(value: object) -> RecordedOrigin | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Stored origin must be an object")
    statements = {key: item for key, item in value.items() if key != "receiver"}
    return RecordedOrigin(
        statements=_parse_origin_input(statements),
        receiver=TypeAdapter(IngestStamp).validate_python(value.get("receiver")),
    )


def _message_facts(message: "Message", *, original_id: str | None = None) -> MessageFacts:
    return MessageFacts(
        message_id=original_id or message.id,
        origin=message.origin,
        legacy_created_at=message.to_dict()["created_at"],
        turn_id=message.turn_id,
        message_kind=message.message_kind,
        source_message_ids=(
            tuple(message.source_message_ids) if message.source_message_ids is not None else None
        ),
    )


def _accepted_origin(message: "Message", *, input_index: int) -> dict:
    facts = _message_facts(message)
    if facts.origin is None:
        raise OriginContractMismatch("Readback lost the recorded origin")
    return {
        "input_index": input_index,
        "message_id": facts.message_id,
        "origin": _origin_to_dict(facts.origin),
        "legacy_created_at": facts.legacy_created_at,
        "turn_id": facts.turn_id,
        "message_kind": facts.message_kind,
        "source_message_ids": list(facts.source_message_ids)
        if facts.source_message_ids is not None
        else None,
    }
