# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from openviking.message.origin import OriginConflict, SelectedMessage
from openviking.session.memory.dataclass import MemoryFile, MemoryOperationSource, ResolvedOperation

SYSTEM_SOURCE_FIELDS = frozenset(
    {
        "source_extractions",
        "source_extraction_id",
        "source_extraction_ids",
        "source_provenance_ambiguous",
        "source_metadata_conflicts",
        "last_update_trace_id",
    }
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, default=asdict)


@dataclass(frozen=True)
class ExtractionOriginView:
    source: MemoryOperationSource
    messages: tuple[SelectedMessage, ...]


@dataclass(frozen=True)
class MemoryOriginView:
    extractions: tuple[ExtractionOriginView, ...]
    ambiguous: bool
    conflicts: tuple[OriginConflict, ...]
    exact_clause_proof: Literal[False] = False
    semantic_withdrawal_established: Literal[False] = False


def _origin_conflicts(sources: list[MemoryOperationSource]) -> tuple[OriginConflict, ...]:
    by_id: dict[str, list[MemoryOperationSource]] = {}
    for source in sources:
        if source.extraction_id:
            by_id.setdefault(source.extraction_id, []).append(source)
    conflicts: dict[str, OriginConflict] = {}

    def compare(extraction_id, message_id, field, values):
        known = {_canonical(value): value for value in values if value is not None}
        if len(known) > 1:
            conflict = OriginConflict(
                extraction_id=extraction_id,
                message_id=message_id,
                field=field,
                variants=tuple(known[key] for key in sorted(known)),
            )
            conflicts[_canonical(conflict)] = conflict

    for extraction_id, variants in by_id.items():
        for field in MemoryOperationSource.model_fields:
            if field != "message_origins":
                compare(extraction_id, None, field, [getattr(source, field) for source in variants])
        messages: dict[str, list] = {}
        for source in variants:
            if source.message_origins is not None:
                for selected in source.message_origins.messages:
                    messages.setdefault(selected.facts.message_id, []).append(selected.facts)
        for message_id, facts in messages.items():
            for field in ("legacy_created_at", "turn_id", "message_kind", "source_message_ids"):
                compare(extraction_id, message_id, field, [getattr(fact, field) for fact in facts])
            origins = [fact.origin for fact in facts if fact.origin is not None]
            compare(extraction_id, message_id, "receiver", [origin.receiver for origin in origins])
            contributors: dict[str, list] = {}
            for origin in origins:
                for contributor in origin.statements.contributors:
                    for ref in contributor.refs:
                        key = _canonical([contributor.locator, ref])
                        group = contributors.setdefault(key, [])
                        group.append(contributor)
            for key, group in contributors.items():
                for name in ("transport_sender", "primary_author"):
                    claims = [
                        getattr(item, name) for item in group if getattr(item, name) is not None
                    ]
                    for field in ("identifier", "scope"):
                        compare(
                            extraction_id,
                            message_id,
                            f"{key}.{name}.{field}",
                            [getattr(claim.actor, field) for claim in claims],
                        )
                compare(
                    extraction_id,
                    message_id,
                    f"{key}.content_class",
                    [item.content_class for item in group if item.content_class != "unknown"],
                )
                times: dict[str, list] = {}
                for item in group:
                    for statement in item.times:
                        times.setdefault(_canonical(statement.basis), []).append(statement)
                for basis, statements in times.items():
                    for field in ("literal", "encoding", "meaning"):
                        compare(
                            extraction_id,
                            message_id,
                            f"{key}.times.{basis}.{field}",
                            [
                                getattr(statement, field)
                                for statement in statements
                                if field != "meaning" or statement.meaning != "unknown"
                            ],
                        )
    return tuple(conflicts[key] for key in sorted(conflicts))


def memory_origin_view(memory: MemoryFile) -> MemoryOriginView:
    sources = [
        MemoryOperationSource.model_validate(value)
        for value in memory.extra_fields.get("source_extractions", [])
    ]
    witnessed = {source.extraction_id for source in sources}
    sources.extend(
        MemoryOperationSource(extraction_id=value)
        for value in sorted(extraction_ids(memory.extra_fields) - witnessed)
    )
    conflicts = _origin_conflicts(sources)
    return MemoryOriginView(
        extractions=tuple(
            ExtractionOriginView(
                source=source,
                messages=source.message_origins.messages if source.message_origins else (),
            )
            for source in sources
        ),
        ambiguous=bool(
            memory.extra_fields.get("source_provenance_ambiguous")
            or memory.extra_fields.get("source_metadata_conflicts")
            or conflicts
        ),
        conflicts=conflicts,
    )


def extraction_ids(fields: dict[str, Any]) -> set[str]:
    ids = set()
    if fields.get("source_extraction_id"):
        ids.add(str(fields["source_extraction_id"]))
    values = fields.get("source_extraction_ids") or []
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    ids.update(str(value) for value in values if value)
    return ids


def operation_source_metadata(
    op: ResolvedOperation, *, old_fields: dict[str, Any] | None = None
) -> dict[str, Any]:
    if old_fields is None:
        old_file = op.old_memory_file_content
        old_fields = old_file.extra_fields if old_file is not None else {}
    sources = [
        MemoryOperationSource.model_validate(value)
        for value in old_fields.get("source_extractions", [])
    ]
    sources.extend(op.sources)
    if op.source is not None:
        sources.append(op.source)
    snapshots = {
        _canonical(source.model_dump(exclude_none=True)): source
        for source in sources
        if source.model_dump(exclude_none=True)
    }
    sources = [snapshots[key] for key in sorted(snapshots)]
    ids = extraction_ids(old_fields) | extraction_ids(op.memory_fields)
    ids.update(source.extraction_id for source in sources if source.extraction_id)
    conflicts = set(old_fields.get("source_metadata_conflicts", []))
    conflicts.update(conflict.extraction_id for conflict in _origin_conflicts(sources))

    metadata: dict[str, Any] = {}
    if sources:
        metadata["source_extractions"] = [
            source.model_dump(exclude_none=True) for source in sources
        ]
    if ids:
        primary = op.source.extraction_id if op.source is not None else None
        primary = (
            primary
            or op.memory_fields.get("source_extraction_id")
            or old_fields.get("source_extraction_id")
        )
        if len(ids) == 1:
            metadata["source_extraction_id"] = next(iter(ids))
        else:
            metadata["source_extraction_ids"] = sorted(ids)
            if primary in ids:
                metadata["source_extraction_id"] = primary
    if op.source_ambiguous or old_fields.get("source_provenance_ambiguous") or conflicts:
        metadata["source_provenance_ambiguous"] = True
    if conflicts:
        metadata["source_metadata_conflicts"] = sorted(conflicts)
    return metadata
