# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""System-owned extraction snapshots, not per-clause evidence."""

import json
from typing import Any

from openviking.session.memory.dataclass import MemoryOperationSource, ResolvedOperation

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
        json.dumps(source.model_dump(exclude_none=True), sort_keys=True): source
        for source in sources
        if source.model_dump(exclude_none=True)
    }
    sources = [snapshots[key] for key in sorted(snapshots)]
    ids = extraction_ids(old_fields) | extraction_ids(op.memory_fields)
    ids.update(source.extraction_id for source in sources if source.extraction_id)
    conflicts = set(old_fields.get("source_metadata_conflicts", []))
    by_id: dict[str, list[MemoryOperationSource]] = {}
    for source in sources:
        if source.extraction_id:
            by_id.setdefault(source.extraction_id, []).append(source)
    for source_id, variants in by_id.items():
        if any(
            len(
                {
                    getattr(source, field)
                    for source in variants
                    if getattr(source, field) is not None
                }
            )
            > 1
            for field in MemoryOperationSource.model_fields
        ):
            conflicts.add(source_id)

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
