# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Numerical OVPack regressions using generated values and pure format helpers."""

import asyncio
import hashlib
import math
import struct
import sys
from array import array
from types import SimpleNamespace

import pytest

from openviking.storage.ovpack import format as ovpack_format
from openviking.storage.ovpack.format import dense_values_bytes
from openviking.storage.ovpack.index import append_index_records, build_manifest
from openviking.storage.ovpack.vectors import build_dense_snapshot_manifest


@pytest.mark.parametrize("compact", [False, True])
def test_dense_bytes_match_prior_encoder(compact):
    values = [
        0.0,
        -0.0,
        1.0,
        -2.5,
        1 / 3,
        2**-149,
        2**-150,
        3 * 2**-150,
        1 + 2**-24,
        1 + 3 * 2**-24,
        float.fromhex("0x1.fffffep+127"),
        math.inf,
        -math.inf,
        math.nan,
    ] * 601
    expected = struct.pack(f"<{len(values)}f", *values)
    source = array("f", values) if compact else values
    before = source.tobytes() if compact else list(source)
    assert dense_values_bytes(source) == expected
    assert (source.tobytes() if compact else source) == before
    assert dense_values_bytes(array("f") if compact else []) == b""


def test_dense_literal_little_endian_and_other_array_type():
    expected = bytes.fromhex("0000803f000020c000000080")
    for source in ([1.0, -2.5, -0.0], array("f", [1.0, -2.5, -0.0]), array("d", [1.0, -2.5, -0.0])):
        assert dense_values_bytes(source) == expected


def test_big_endian_branch_swaps_a_copy(monkeypatch):
    source = array("f", [1.0, -2.5])
    before = source.tobytes()
    swapped = array("f", source)
    swapped.byteswap()
    # Exercise the host-order branch; this does not emulate a big-endian CPU.
    monkeypatch.setattr(ovpack_format.sys, "byteorder", "big")
    assert dense_values_bytes(source) == swapped.tobytes()
    assert source.tobytes() == before


@pytest.mark.parametrize(
    "value", [1e39, -1e39, float.fromhex("0x1.ffffffp+127"), "1", None, complex(1, 0), 10**400]
)
def test_list_encoder_preserves_invalid_input_errors(value):
    with pytest.raises(Exception) as prior:
        struct.pack("<f", value)
    with pytest.raises(type(prior.value)):
        dense_values_bytes([value])


def test_offsets_counts_checksum_and_input_ownership():
    records = [
        {"level": 0, "_dense_vector": [1.0, -2.5]},
        {"level": 1, "_dense_vector": [1 / 3, -0.0]},
        {"level": 2, "_dense_vector": [True, 1.0]},
    ]
    dense = array("f")
    index = []
    append_index_records(index, dense, "generated.txt", "file", records)
    assert [record.get("vector") for record in index] == [
        {"dense": {"offset": 0, "dimensions": 2}},
        {"dense": {"offset": 2, "dimensions": 2}},
        None,
    ]
    expected = struct.pack("<4f", 1.0, -2.5, 1 / 3, -0.0)
    payload, metadata = build_dense_snapshot_manifest(
        index, dense, SimpleNamespace(hybrid=None, dense=None)
    )
    assert payload == expected
    assert metadata == {
        "path": "_ovpack/dense.f32",
        "count": 2,
        "dtype": "float32",
        "byte_order": "little",
        "dimensions": 2,
        "sha256": hashlib.sha256(expected).hexdigest(),
        "embedding": {"dimensions": 2},
    }
    assert records[0]["_dense_vector"] == [1.0, -2.5]
    assert records[1]["_dense_vector"] == [1 / 3, -0.0]
    legacy_index, legacy_dense = [], []
    append_index_records(legacy_index, legacy_dense, "generated.txt", "file", records)
    assert legacy_index == index
    assert dense_values_bytes(legacy_dense) == expected


def test_compact_accumulator_rejects_float32_overflow():
    with pytest.raises(OverflowError):
        append_index_records([], array("f"), "generated.txt", "file", [{"_dense_vector": [1e39]}])


def test_manifest_accumulator_memory_scales_as_float32():
    _, index, dense = asyncio.run(build_manifest(None, None, "viking://", "generated", [], None))
    assert isinstance(dense, array)
    assert dense.itemsize == 4
    empty_size = sys.getsizeof(dense)
    record = {"_dense_vector": [float(i) / 4096 for i in range(4096)]}
    sizes = []
    for count in range(1, 33):
        append_index_records(index, dense, f"generated-{count}", "file", [record])
        if count in (16, 32):
            sizes.append(sys.getsizeof(dense) - empty_size)
    assert len(dense) == 32 * 4096
    assert index[-1]["vector"]["dense"] == {"offset": 31 * 4096, "dimensions": 4096}
    assert sizes[1] <= len(dense) * 4 * 1.15
    assert 1.8 <= sizes[1] / sizes[0] <= 2.2
