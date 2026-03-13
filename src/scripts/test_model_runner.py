#!/usr/bin/env python3
"""
Unit tests for GPUModelRunner lifecycle.

Tests (run in order, each building on the last):
  1. model_load        – weights load; model is on the right device in eval mode
  2. kv_cache_memory   – determine_available_memory() returns a sane value
  3. kv_cache_sizing   – compute_num_gpu_blocks() produces valid (num_blocks, max_seq_len)
  4. kv_cache_init     – initialize_kv_cache() allocates K/V buffers with correct shapes
  5. execute_model     – a single prefill forward pass returns the right output type/shape
  6. kv_cache_shapes   – K/V tensors on every attention block match expected geometry

Usage (from project root):
    python -m src.scripts.test_model_runner
"""

import sys
import os
import time
import traceback

# --------------------------------------------------------------------------
# Path setup (litgpt lives in a submodule)
# --------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "litgpt"))

from pathlib import Path

import torch
from typing import Any

from src.config.vllm import BLOCK_SIZE, VllmConfig
from src.worker.model_runner import GPUModelRunner
from src.worker.model_input import SchedulerOutput

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
CHECKPOINT_DIR = Path(ROOT) / "checkpoints/meta-llama/Meta-Llama-3.1-8B-Instruct"

# max_num_seqs=1: Llama-3.1-8B in bf16 (~14.9 GB) nearly fills a 16 GB GPU,
# leaving very little room for KV cache.  A single sequence keeps the
# per-token KV overhead minimal so the memory-profiling step succeeds.
# Increase to 2–4 on GPUs with 24 GB+ VRAM.
VLLM_CONFIG = VllmConfig(
    checkpoint_dir=CHECKPOINT_DIR,
    max_num_seqs=1,
    gpu_memory_utilization=0.9,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_results: list[tuple[str, bool, str]] = []


def run(name: str, fn):
    """Run a single test, record result, never abort the suite."""
    t0 = time.perf_counter()
    try:
        fn()
        elapsed = time.perf_counter() - t0
        _results.append((name, True, f"{elapsed:.2f}s"))
        print(f"  {PASS}  {name}  ({elapsed:.2f}s)")
    except Exception:
        elapsed = time.perf_counter() - t0
        _results.append((name, False, f"{elapsed:.2f}s"))
        print(f"  {FAIL}  {name}  ({elapsed:.2f}s)")
        traceback.print_exc()


# --------------------------------------------------------------------------
# Shared runner (loaded once, reused across tests)
# --------------------------------------------------------------------------
_runner: GPUModelRunner | None = None
_num_blocks: int = 0
_max_seq_length: int = 0


def _get_runner() -> GPUModelRunner:
    global _runner
    if _runner is None:
        _runner = GPUModelRunner(VLLM_CONFIG)
    return _runner


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_model_load():
    runner = _get_runner()
    runner.load_model()

    assert runner.model is not None, "model is None after load_model()"
    assert not runner.model.training, "model should be in eval mode"
    assert runner.device is not None, "device not set"

    device_type = runner.device.type
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert device_type == expected, f"expected device {expected}, got {device_type}"

    # Spot-check: embedding weight lives on the right device
    model: Any = runner.model
    wte_device = next(model.transformer.wte.parameters()).device.type
    assert wte_device == device_type, f"weight on {wte_device}, expected {device_type}"

    print(f"    device={runner.device}, "
          f"n_layer={runner.model.config.n_layer}, "
          f"n_query_groups={runner.model.config.n_query_groups}, "
          f"head_size={runner.model.config.head_size}")


def test_kv_cache_memory():
    runner = _get_runner()
    available = runner.determine_available_memory()

    assert isinstance(available, int), f"expected int, got {type(available)}"
    assert available >= 0, f"available memory is {available} bytes — negative"

    if available == 0:
        print("    available_for_kv = 0 bytes  "
              "(model fills GPU; 1-block minimum KV cache will be used)")
    else:
        print(f"    available_for_kv = {available / 1024**3:.2f} GiB")


def test_kv_cache_sizing():
    global _num_blocks, _max_seq_length
    runner = _get_runner()

    available = runner.determine_available_memory()
    num_blocks, max_seq_length = runner.compute_num_gpu_blocks(available)

    assert num_blocks >= 1, f"num_gpu_blocks={num_blocks} < 1"
    assert max_seq_length >= BLOCK_SIZE, (
        f"max_seq_length={max_seq_length} < BLOCK_SIZE={BLOCK_SIZE}"
    )
    assert max_seq_length % BLOCK_SIZE == 0, (
        f"max_seq_length={max_seq_length} not a multiple of BLOCK_SIZE={BLOCK_SIZE}"
    )
    assert max_seq_length <= runner.model.config.block_size, (
        f"max_seq_length={max_seq_length} exceeds model block_size={runner.model.config.block_size}"
    )
    assert num_blocks == max_seq_length // BLOCK_SIZE

    _num_blocks = num_blocks
    _max_seq_length = max_seq_length
    print(f"    num_gpu_blocks={num_blocks}, max_seq_length={max_seq_length}")


def test_kv_cache_init():
    runner = _get_runner()
    assert _max_seq_length > 0, "run test_kv_cache_sizing first"

    runner.initialize_kv_cache(_num_blocks, _max_seq_length)

    # Verify every transformer block has a KV cache with the right shapes
    cfg = runner.model.config
    expected_batch = VLLM_CONFIG.max_num_seqs
    expected_groups = cfg.n_query_groups
    expected_head = cfg.head_size

    model: Any = runner.model
    blocks: list[Any] = list(model.transformer.h)

    n_blocks_with_cache = 0
    for block in blocks:
        kv = block.attn.kv_cache
        assert kv is not None, "kv_cache not set on attention block"
        # V shape: (batch_size, n_query_groups, max_seq_length, head_size)
        assert kv.v.shape[0] == expected_batch, (
            f"V batch dim {kv.v.shape[0]} != {expected_batch}"
        )
        assert kv.v.shape[1] == expected_groups, (
            f"V groups dim {kv.v.shape[1]} != {expected_groups}"
        )
        assert kv.v.shape[2] == _max_seq_length, (
            f"V seq dim {kv.v.shape[2]} != {_max_seq_length}"
        )
        assert kv.v.shape[3] == expected_head, (
            f"V head dim {kv.v.shape[3]} != {expected_head}"
        )
        n_blocks_with_cache += 1

    assert n_blocks_with_cache == cfg.n_layer, (
        f"only {n_blocks_with_cache}/{cfg.n_layer} blocks have a KV cache"
    )

    # mask_cache should also be set
    assert model.mask_cache is not None, "mask_cache not set after set_kv_cache"

    # Spot-check total VRAM used for KV cache
    kv0 = blocks[0].attn.kv_cache
    bytes_per_layer = (kv0.k.nelement() * kv0.k.element_size()
                       + kv0.v.nelement() * kv0.v.element_size())
    total_kv_bytes = bytes_per_layer * cfg.n_layer

    print(f"    KV cache shapes — V: {tuple(kv0.v.shape)}, K: {tuple(kv0.k.shape)}")
    print(f"    KV cache total = {total_kv_bytes / 1024**3:.2f} GiB "
          f"across {cfg.n_layer} layers")


def test_execute_model_prefill():
    """Single-sequence prefill: 8 tokens → 1 sampled next token."""
    runner = _get_runner()
    assert runner.model.mask_cache is not None, "initialize kv cache first"

    prompt_ids = [128000, 9906, 1917, 0, 128009, 128006, 78191, 128007, 271]
    # (BOS) "Hello world !" + special tokens for Llama-3 instruct format
    T = len(prompt_ids)
    positions = list(range(T))

    sched_out = SchedulerOutput(
        input_ids=[prompt_ids],
        positions=[positions],
    )

    output = runner.execute_model(sched_out)

    assert len(output.sampled_token_ids) == 1, (
        f"expected 1 sampled token, got {len(output.sampled_token_ids)}"
    )
    token_id = output.sampled_token_ids[0]
    assert isinstance(token_id, int), f"token_id should be int, got {type(token_id)}"
    vocab_size = runner.model.config.padded_vocab_size or 0
    assert vocab_size > 0, "padded_vocab_size not set on model config"
    assert 0 <= token_id < vocab_size, (
        f"token_id={token_id} out of vocab range [0, {vocab_size})"
    )
    print(f"    sampled next token id = {token_id}  (vocab_size={vocab_size})")


def test_execute_model_decode():
    """Decode step: pass a single new token at position T after prefill."""
    runner = _get_runner()
    assert runner.model.mask_cache is not None, "initialize kv cache first"

    # Simulate one decode step: single token at position 9 (after 9-token prefill)
    new_token = 14524  # arbitrary non-special token id
    position = 9

    sched_out = SchedulerOutput(
        input_ids=[[new_token]],
        positions=[[position]],
    )

    output = runner.execute_model(sched_out)

    assert len(output.sampled_token_ids) == 1
    token_id = output.sampled_token_ids[0]
    vocab_size = runner.model.config.padded_vocab_size or 0
    assert vocab_size > 0 and 0 <= token_id < vocab_size
    print(f"    decode step sampled token id = {token_id}")


def test_batch_execute_model():
    """
    Multiple sequences in one batch.  Skipped when max_num_seqs == 1 because
    the KV cache is dimensioned to that batch size and can't handle more.
    """
    runner = _get_runner()
    assert runner.model.mask_cache is not None, "initialize kv cache first"

    if VLLM_CONFIG.max_num_seqs < 2:
        print(f"    SKIP — max_num_seqs={VLLM_CONFIG.max_num_seqs} < 2; "
              "increase max_num_seqs for batch testing (requires more VRAM)")
        return

    # Two sequences of different lengths — shorter one will be right-padded
    seq_a = [128000, 9906, 1917]        # 3 tokens
    seq_b = [128000, 9906, 1917, 0, 1]  # 5 tokens

    sched_out = SchedulerOutput(
        input_ids=[seq_a, seq_b],
        positions=[list(range(len(seq_a))), list(range(len(seq_b)))],
    )

    output = runner.execute_model(sched_out)
    assert len(output.sampled_token_ids) == 2, (
        f"expected 2 tokens (one per sequence), got {len(output.sampled_token_ids)}"
    )
    print(f"    batch sampled token ids = {output.sampled_token_ids}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    print(f"\n{'='*60}")
    print(f"  GPUModelRunner tests — {CHECKPOINT_DIR.name}")
    print(f"  BLOCK_SIZE={BLOCK_SIZE}, max_num_seqs={VLLM_CONFIG.max_num_seqs}")
    print(f"{'='*60}\n")

    run("1. model_load",           test_model_load)
    run("2. kv_cache_memory",      test_kv_cache_memory)
    run("3. kv_cache_sizing",      test_kv_cache_sizing)
    run("4. kv_cache_init",        test_kv_cache_init)
    run("5. execute_model_prefill", test_execute_model_prefill)
    run("6. execute_model_decode",  test_execute_model_decode)
    run("7. batch_execute_model",   test_batch_execute_model)

    print(f"\n{'='*60}")
    passed = sum(ok for _, ok, _ in _results)
    total = len(_results)
    print(f"  Result: {passed}/{total} passed")
    print(f"{'='*60}\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
