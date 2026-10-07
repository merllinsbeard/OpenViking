# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Generated numerical controls. ZIP reads are real; the upsert sink is a test double."""

import asyncio
import io
import json
import math
import struct
import sys
import zipfile
from array import array
from types import SimpleNamespace

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.ovpack import vectors as vector_helpers
from openviking.storage.ovpack.format import dense_values_bytes
from openviking.storage.ovpack.vectors import (
    build_dense_snapshot_manifest,
    read_dense_vectors,
    restore_vector_snapshot,
)
from openviking_cli.session.user_id import UserIdentifier


def _decode(payload, records, dimensions):
    _, dense_info = build_dense_snapshot_manifest(
        [{"vector": {"dense": {"offset": 0, "dimensions": dimensions}}}],
        [0.0] * dimensions,
        SimpleNamespace(hybrid=None, dense=None),
    )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("generated/_ovpack/dense.f32", payload)
    with zipfile.ZipFile(archive) as zf:
        return read_dense_vectors(zf, {"index": {"dense": dense_info}}, "generated", records)


def test_decode_exact_offsets_and_float32_bits():
    words = [
        0x00000000,
        0x80000000,
        0x00000001,
        0x007FFFFF,
        0x00800000,
        0x3EAAAAAB,
        0x7F7FFFFF,
        0xFF7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7FC01234,
        0xFFC01234,
    ]
    payload = struct.pack("<12I", *words)
    records = [
        {"record_id": "second", "vector": {"dense": {"offset": 6, "dimensions": 6}}},
        {"record_id": "first", "vector": {"dense": {"offset": 0, "dimensions": 6}}},
        {"record_id": "text-only"},
    ]
    decoded = _decode(payload, records, 6)
    assert set(decoded) == {"first", "second"}
    for record in records[:2]:
        ref = record["vector"]["dense"]
        vector = decoded[record["record_id"]]
        expected = payload[ref["offset"] * 4 : (ref["offset"] + 6) * 4]
        assert dense_values_bytes(vector) == expected
        prior = struct.unpack("<6f", expected)
        assert all(
            (math.isnan(a) and math.isnan(b)) or a == b for a, b in zip(vector, prior, strict=True)
        )
        assert struct.pack("<6f", *list(vector)) == struct.pack("<6f", *prior)
    assert payload == struct.pack("<12I", *words)


@pytest.mark.parametrize("offset,dimensions", [(1, 2), (0, -1), (-4, 2)])
def test_decode_preserves_invalid_reference_errors(offset, dimensions):
    payload = struct.pack("<2f", 1.0, 2.0)
    with pytest.raises((struct.error, ValueError)) as prior:
        struct.unpack_from(f"<{dimensions}f", payload, offset * 4)
    with pytest.raises(type(prior.value)):
        _decode(
            payload,
            [
                {
                    "record_id": "bad",
                    "vector": {"dense": {"offset": offset, "dimensions": dimensions}},
                }
            ],
            2,
        )


def test_decode_negative_offset_retains_struct_semantics():
    decoded = _decode(
        struct.pack("<2f", 1.0, 2.0),
        [{"record_id": "last", "vector": {"dense": {"offset": -1, "dimensions": 1}}}],
        1,
    )
    assert list(decoded["last"]) == [2.0]


def test_decode_big_endian_branch(monkeypatch):
    payload = bytes.fromhex("0000803f000020c0")
    monkeypatch.setattr(vector_helpers.sys, "byteorder", "big")
    decoded = _decode(
        payload, [{"record_id": "one", "vector": {"dense": {"offset": 0, "dimensions": 2}}}], 2
    )
    # Branch control on a little-endian CPU, rather than a big-endian platform claim.
    expected = array("f")
    expected.frombytes(payload)
    expected.byteswap()
    assert decoded["one"].tobytes() == expected.tobytes()


def test_decoded_storage_scales_as_float32():
    sizes = []
    for count in (16, 32):
        payload = struct.pack("<4096f", *(i / 4096 for i in range(4096))) * count
        records = [
            {"record_id": str(i), "vector": {"dense": {"offset": i * 4096, "dimensions": 4096}}}
            for i in range(count)
        ]
        decoded = _decode(payload, records, 4096)
        assert all(
            isinstance(vector, array) and vector.itemsize == 4 for vector in decoded.values()
        )
        compact_size = sum(sys.getsizeof(vector) for vector in decoded.values())
        legacy = [list(struct.unpack_from("<4096f", payload, i * 4096 * 4)) for i in range(count)]
        legacy_size = sum(sys.getsizeof(v) + sum(sys.getsizeof(x) for x in v) for v in legacy)
        assert compact_size < len(payload) * 1.10
        assert legacy_size / compact_size > 7
        sizes.append(compact_size)
    assert 1.95 <= sizes[1] / sizes[0] <= 2.05


def test_restore_converts_only_upsert_payload_and_keeps_legacy_lists():
    class RecordingSink:
        async def upsert(self, payload, *, ctx):
            assert isinstance(payload["vector"], list)
            self.payloads.append(payload)
            json.dumps(payload)

    sink = RecordingSink()
    sink.payloads = []
    compact = array("f", [1.0, -0.0])
    legacy = [2.0, -2.5]
    vectors = {"compact": compact, "legacy": legacy}
    records = [
        {"record_id": key, "path": "generated.txt", "level": level}
        for level, key in enumerate(vectors)
    ]
    ctx = RequestContext(UserIdentifier("generated", "generated"), Role.ADMIN)
    asyncio.run(
        restore_vector_snapshot(
            sink,
            "viking://resources/generated",
            records,
            vectors,
            {"generated.txt": {"kind": "file"}},
            ctx,
        )
    )
    assert [p["vector"] for p in sink.payloads] == [[1.0, -0.0], [2.0, -2.5]]
    assert sink.payloads[1]["vector"] is legacy
    assert vectors["compact"] is compact
    assert dense_values_bytes(compact) == bytes.fromhex("0000803f00000080")
    assert all(p["account_id"] == "generated" for p in sink.payloads)
