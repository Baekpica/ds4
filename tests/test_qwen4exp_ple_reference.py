#!/usr/bin/env python3
"""Cross-check DS4 PLE hashing and SSD rows against pinned Qwen sources."""

from __future__ import annotations

import argparse
import bisect
import ctypes
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from safetensors import safe_open

N_HEADS = 16
ROW_BYTES = 320
EOS = 248044
VOCAB = 248320
REFERENCE_SHA256 = "77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459"
REFERENCE_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/transformers/models/"
    "qwen4_exp/modeling_qwen4_exp.py"
)


class HashConfig(ctypes.Structure):
    _fields_ = [
        ("unigram_vocab_size", ctypes.c_uint32),
        ("eos_token_id", ctypes.c_uint32),
        ("layer_multipliers", ctypes.c_uint64 * 3),
        ("head_vocab_sizes", ctypes.c_uint64 * 16),
        ("head_offsets", ctypes.c_uint64 * 16),
    ]


class HashState(ctypes.Structure):
    _fields_ = [("previous", ctypes.c_int64 * 2)]


class Layout(ctypes.Structure):
    _fields_ = [
        ("format_version", ctypes.c_uint32),
        ("alignment_bytes", ctypes.c_uint32),
        ("row_stride_bytes", ctypes.c_uint32),
        ("embedding_row_dimension", ctypes.c_uint32),
        ("logical_part_count", ctypes.c_uint32),
        ("physical_file_count", ctypes.c_uint32),
        ("usable_vocabulary_rows", ctypes.c_uint64),
        ("padded_vocabulary_rows", ctypes.c_uint64),
        ("total_payload_bytes", ctypes.c_uint64),
        ("total_file_bytes", ctypes.c_uint64),
        ("cache_bytes", ctypes.c_size_t),
        ("cache_slots", ctypes.c_uint32),
        ("worker_count", ctypes.c_uint32),
        ("direct_io_file_count", ctypes.c_uint32),
    ]


class Stats(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "row_lookups",
        "logical_bytes",
        "page_requests",
        "cache_hits",
        "cache_inflight_hits",
        "cache_misses",
        "cache_evictions",
        "prefetch_dropped",
        "read_operations",
        "physical_bytes",
        "read_errors",
        "wait_samples",
        "wait_nanoseconds_total",
        "wait_nanoseconds_max",
    )]


def bind_library(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    lib.ds4_ple_hash_state_reset.argtypes = [
        ctypes.POINTER(HashState), ctypes.POINTER(HashConfig)
    ]
    lib.ds4_ple_hash_rows.argtypes = [
        ctypes.POINTER(HashConfig),
        ctypes.POINTER(HashState),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.ds4_ple_hash_rows.restype = ctypes.c_bool
    lib.ds4_ple_store_open.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_bool,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.ds4_ple_store_open.restype = ctypes.c_void_p
    lib.ds4_ple_store_close.argtypes = [ctypes.c_void_p]
    lib.ds4_ple_store_layout.argtypes = [ctypes.c_void_p]
    lib.ds4_ple_store_layout.restype = ctypes.POINTER(Layout)
    lib.ds4_ple_store_hash_config.argtypes = [ctypes.c_void_p]
    lib.ds4_ple_store_hash_config.restype = ctypes.POINTER(HashConfig)
    lib.ds4_ple_store_prefetch_rows.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.ds4_ple_store_prefetch_rows.restype = ctypes.c_bool
    lib.ds4_ple_store_read_row.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.ds4_ple_store_read_row.restype = ctypes.c_bool
    lib.ds4_ple_store_get_stats.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(Stats)
    ]
    return lib


def shift_right_ignore_eos(token_ids: torch.Tensor, shift: int) -> torch.Tensor:
    if shift == 0:
        return token_ids
    batch_size, seq_len = token_ids.shape
    positions = torch.arange(seq_len, dtype=torch.long)
    eos_positions = torch.where(token_ids == EOS, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat(
        [eos_positions.new_full((batch_size, 1), -1),
         previous_eos_inclusive[:, :-1]],
        dim=1,
    )
    segment_start = previous_eos + 1
    position_in_segment = positions.unsqueeze(0) - segment_start
    source_positions = positions - shift
    gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(
        batch_size, -1
    )
    shifted = token_ids.gather(dim=1, index=gather_positions)
    valid = (
        (position_in_segment >= shift)
        & (source_positions.unsqueeze(0) >= 0)
    )
    return torch.where(valid, shifted, token_ids.new_full((), EOS))


def reference_rows(tokens: list[int], config: HashConfig) -> list[int]:
    input_ids = torch.tensor([tokens], dtype=torch.long)
    previous = torch.full((1, 2), EOS, dtype=torch.long)
    history = torch.cat([previous, input_ids], dim=-1)
    shifted = [shift_right_ignore_eos(history, i) for i in range(3)]
    multipliers = torch.tensor(
        list(config.layer_multipliers), dtype=torch.long
    )
    vocab_sizes = torch.tensor(
        list(config.head_vocab_sizes), dtype=torch.long
    )
    offsets = torch.tensor(list(config.head_offsets), dtype=torch.long)
    blocks = []
    for ngram in (2, 3):
        start = (ngram - 2) * 8
        mixed = shifted[0] * multipliers[0]
        for position in range(1, ngram):
            mixed = torch.bitwise_xor(
                mixed, shifted[position] * multipliers[position]
            )
        blocks.append(
            torch.remainder(mixed.unsqueeze(-1), vocab_sizes[start:start + 8])
            + offsets[start:start + 8]
        )
    return torch.cat(blocks, dim=-1)[:, -len(tokens):].reshape(-1).tolist()


def c_rows(
    lib: ctypes.CDLL,
    config: ctypes.POINTER(HashConfig),
    state: HashState,
    tokens: list[int],
) -> tuple[list[int], HashState]:
    input_array = (ctypes.c_int64 * len(tokens))(*tokens)
    output = (ctypes.c_uint64 * (len(tokens) * N_HEADS))()
    error = ctypes.create_string_buffer(512)
    ok = lib.ds4_ple_hash_rows(
        config,
        ctypes.byref(state),
        input_array,
        len(tokens),
        output,
        error,
        len(error),
    )
    if not ok:
        raise AssertionError(error.value.decode())
    return list(output), state


def test_hashes(
    lib: ctypes.CDLL,
    config: ctypes.POINTER(HashConfig),
    rng: random.Random,
) -> dict[str, int]:
    sequence_count = 512
    token_count = 0
    row_count = 0
    for case in range(sequence_count):
        length = rng.randint(1, 128)
        tokens = [rng.randrange(VOCAB) for _ in range(length)]
        for i in range(length):
            if rng.random() < 0.08:
                tokens[i] = EOS
        expected = reference_rows(tokens, config.contents)

        state = HashState()
        lib.ds4_ple_hash_state_reset(ctypes.byref(state), config)
        actual, _ = c_rows(lib, config, state, tokens)
        if actual != expected:
            mismatch = next(
                i for i, pair in enumerate(zip(actual, expected))
                if pair[0] != pair[1]
            )
            raise AssertionError(
                f"whole-sequence hash mismatch in case {case}, row {mismatch}"
            )

        state = HashState()
        lib.ds4_ple_hash_state_reset(ctypes.byref(state), config)
        chunked: list[int] = []
        at = 0
        while at < length:
            width = min(length - at, rng.randint(1, 17))
            rows, state = c_rows(
                lib, config, state, tokens[at:at + width]
            )
            chunked.extend(rows)
            at += width
        if chunked != expected:
            raise AssertionError(f"chunked hash mismatch in case {case}")
        token_count += length
        row_count += len(expected)

    long_tokens = [rng.randrange(VOCAB) for _ in range(4096)]
    for i in range(0, len(long_tokens), 97):
        long_tokens[i] = EOS
    expected = reference_rows(long_tokens, config.contents)
    state = HashState()
    lib.ds4_ple_hash_state_reset(ctypes.byref(state), config)
    actual: list[int] = []
    at = 0
    for width in (1, 2, 31, 7, 511, 3, 1024, 2517):
        rows, state = c_rows(
            lib, config, state, long_tokens[at:at + width]
        )
        actual.extend(rows)
        at += width
    if actual != expected:
        raise AssertionError("long chunk-boundary hash mismatch")
    return {
        "random_sequences": sequence_count,
        "random_tokens": token_count,
        "random_row_ids": row_count,
        "long_tokens": len(long_tokens),
        "long_row_ids": len(expected),
    }


def locate_part(parts: list[dict], starts: list[int], row: int) -> dict:
    index = bisect.bisect_right(starts, row) - 1
    if index < 0:
        raise AssertionError(f"row {row} precedes the first logical part")
    part = parts[index]
    if row >= part["global_row_start"] + part["rows"]:
        raise AssertionError(f"row {row} falls in a logical gap")
    return part


def c_read_row(
    lib: ctypes.CDLL, store: int, row: int
) -> bytes:
    output = (ctypes.c_uint8 * ROW_BYTES)()
    error = ctypes.create_string_buffer(512)
    if not lib.ds4_ple_store_read_row(
        store, row, output, len(output), error, len(error)
    ):
        raise AssertionError(error.value.decode())
    return bytes(output)


def test_sidecar_rows(
    lib: ctypes.CDLL,
    store: int,
    artifact_root: Path,
    source_root: Path,
    rng: random.Random,
) -> dict[str, int | float]:
    manifest = json.loads(
        (artifact_root / "ple" / "ple-manifest.json").read_text()
    )
    parts = manifest["logical_parts"]
    starts = [part["global_row_start"] for part in parts]

    selected: set[int] = set()
    for part in parts:
        start = part["global_row_start"]
        selected.update((start, start + 1, start + 12, start + part["rows"] - 1))
    while len(selected) < 1024:
        selected.add(rng.randrange(manifest["usable_vocabulary_rows"]))
    rows = sorted(
        row for row in selected
        if row < manifest["usable_vocabulary_rows"]
    )
    row_array = (ctypes.c_uint64 * len(rows))(*rows)
    error = ctypes.create_string_buffer(512)
    if not lib.ds4_ple_store_prefetch_rows(
        store, row_array, len(rows), error, len(error)
    ):
        raise AssertionError(error.value.decode())

    grouped: dict[str, list[tuple[int, dict]]] = {}
    for row in rows:
        part = locate_part(parts, starts, row)
        grouped.setdefault(part["source_shard"], []).append((row, part))

    started = time.perf_counter()
    checked = 0
    for shard, shard_rows in grouped.items():
        with safe_open(source_root / shard, framework="pt", device="cpu") as handle:
            for row, part in shard_rows:
                local = row - part["global_row_start"]
                expected_tensor = handle.get_slice(part["source_name"])[
                    local:local + 1
                ]
                expected = (
                    expected_tensor.contiguous()
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                )
                actual = c_read_row(lib, store, row)
                if actual != expected:
                    raise AssertionError(
                        f"BF16 sidecar mismatch at global row {row}"
                    )
                checked += 1

    hot_rows = rows[-32:]
    for row in hot_rows:
        c_read_row(lib, store, row)
    first = Stats()
    lib.ds4_ple_store_get_stats(store, ctypes.byref(first))
    for row in hot_rows:
        c_read_row(lib, store, row)
    second = Stats()
    lib.ds4_ple_store_get_stats(store, ctypes.byref(second))
    if second.cache_hits <= first.cache_hits:
        raise AssertionError("hot-row reread did not produce cache hits")
    if second.read_errors:
        raise AssertionError(f"sidecar cache reported {second.read_errors} read errors")
    mean_wait = (
        second.wait_nanoseconds_total / second.wait_samples
        if second.wait_samples
        else 0.0
    )
    return {
        "source_rows_checked": checked,
        "source_shards_opened": len(grouped),
        "elapsed_seconds": time.perf_counter() - started,
        "row_lookups": second.row_lookups,
        "logical_bytes": second.logical_bytes,
        "cache_hits": second.cache_hits,
        "cache_inflight_hits": second.cache_inflight_hits,
        "cache_misses": second.cache_misses,
        "cache_evictions": second.cache_evictions,
        "prefetch_dropped": second.prefetch_dropped,
        "read_operations": second.read_operations,
        "physical_bytes": second.physical_bytes,
        "wait_samples": second.wait_samples,
        "wait_nanoseconds_mean": mean_wait,
        "wait_nanoseconds_total": second.wait_nanoseconds_total,
        "wait_nanoseconds_max": second.wait_nanoseconds_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()

    digest = hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest()
    if digest != REFERENCE_SHA256:
        raise SystemExit(
            f"pinned Transformers implementation SHA mismatch: {digest}"
        )

    lib = bind_library(args.library.resolve())
    error = ctypes.create_string_buffer(512)
    store = lib.ds4_ple_store_open(
        str(args.artifact_root.resolve()).encode(),
        b"ple/ple-manifest.json",
        16 * 1024 * 1024,
        8,
        True,
        error,
        len(error),
    )
    if not store:
        raise SystemExit(f"cannot open PLE store: {error.value.decode()}")
    try:
        layout = lib.ds4_ple_store_layout(store).contents
        config = lib.ds4_ple_store_hash_config(store)
        rng = random.Random(0x51454E38)
        hash_result = test_hashes(lib, config, rng)
        sidecar_result = test_sidecar_rows(
            lib, store, args.artifact_root, args.source_root, rng
        )
        report = {
            "status": "passed",
            "reference_implementation_sha256": digest,
            "hash": hash_result,
            "sidecar": sidecar_result,
            "layout": {
                "cache_bytes": layout.cache_bytes,
                "cache_slots": layout.cache_slots,
                "direct_io_files": layout.direct_io_file_count,
                "physical_files": layout.physical_file_count,
                "usable_vocabulary_rows": layout.usable_vocabulary_rows,
            },
        }
        print(json.dumps(report, sort_keys=True))
    finally:
        lib.ds4_ple_store_close(store)


if __name__ == "__main__":
    main()
