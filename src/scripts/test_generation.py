#!/usr/bin/env python3
"""
End-to-end generation test.

Exercises the full stack:
  AsyncEngine → UniProcExecutor → GPUModelRunner → litgpt GPT

Runs prefill + decode for each prompt using the Llama 3.1 tokenizer.
Prompts are processed one at a time (batch_size=1 per KV cache reset) so
the script works even on GPUs where the model fills most available VRAM and
only a minimal KV cache can be allocated.

Usage (from project root):
    python -m src.scripts.test_generation
    python -m src.scripts.test_generation --prompt "Tell me a joke" --max-tokens 64
    python -m src.scripts.test_generation --no-chat-template --prompt "1 + 1 ="
"""

import sys
import os
import argparse
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "litgpt"))

from pathlib import Path

import torch

from src.config.vllm import VllmConfig
from src.engine.async_engine import AsyncEngine
from src.executor.uniproc_executor import UniProcExecutor
from src.worker.model_input import SchedulerOutput

from litgpt.tokenizer import Tokenizer  # type: ignore[import-untyped]

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
CHECKPOINT_DIR = Path(ROOT) / "checkpoints/meta-llama/Meta-Llama-3.1-8B-Instruct"

DEFAULT_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
]
# Short prompts for GPUs where the model fills available VRAM and only a
# minimal KV cache (< 32 tokens) can be allocated.
SHORT_PROMPTS = ["Hello", "1 + 1 ="]

DEFAULT_MAX_TOKENS = 64
# max_num_seqs=1: Llama-3.1-8B bf16 (~14.9 GB) nearly fills a 16 GB GPU.
# Raise to 2-4 on GPUs with 24 GB+ VRAM.
DEFAULT_MAX_NUM_SEQS = 1


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _build_llama3_prompt(text: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{text}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _generate_one(
    model_runner,
    token_ids: list[int],
    max_new_tokens: int,
    max_seq_length: int,
    eos_id: int,
) -> tuple[list[int], float, float]:
    """
    Prefill + decode loop for a single sequence.

    The KV cache is cleared and re-initialised before each call so sequences
    don't interfere with each other.

    Returns (generated_token_ids, prefill_ms, decode_tok_per_sec).
    """
    T = len(token_ids)
    budget = max_seq_length - T
    if budget <= 0:
        print(f"    WARNING: prompt ({T} tokens) >= max_seq_length "
              f"({max_seq_length}).  Skipping.")
        return [], 0.0, 0.0

    max_steps = min(max_new_tokens, budget)

    # Reset KV cache state for this sequence (batch_size=1)
    model_runner.model.clear_kv_cache()
    model_runner.model.set_kv_cache(
        batch_size=1,
        max_seq_length=max_seq_length,
        device=model_runner.device,
    )

    # Prefill: process all prompt tokens at once
    t0 = time.perf_counter()
    out = model_runner.execute_model(
        SchedulerOutput(input_ids=[token_ids], positions=[list(range(T))])
    )
    prefill_ms = (time.perf_counter() - t0) * 1000

    generated = [out.sampled_token_ids[0]]
    pos = T

    # Decode loop: one new token per step
    t_dec = time.perf_counter()
    for _ in range(max_steps - 1):
        if generated[-1] == eos_id:
            break
        out = model_runner.execute_model(
            SchedulerOutput(input_ids=[[generated[-1]]], positions=[[pos]])
        )
        generated.append(out.sampled_token_ids[0])
        pos += 1
    elapsed_dec = time.perf_counter() - t_dec

    tok_per_sec = len(generated) / elapsed_dec if elapsed_dec > 0 else 0.0

    if generated and generated[-1] == eos_id:
        generated = generated[:-1]

    return generated, prefill_ms, tok_per_sec


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="mini-vllm generation test")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--short", action="store_true",
                        help="Use short prompts (fits in tiny KV cache on 16 GB GPUs)")
    args = parser.parse_args()

    if args.prompt:
        prompts = [args.prompt]
    elif args.short:
        prompts = SHORT_PROMPTS
    else:
        prompts = DEFAULT_PROMPTS

    print(f"\n{'='*60}")
    print("  mini-vllm generation test")
    print(f"  checkpoint  : {CHECKPOINT_DIR.name}")
    print(f"  prompts     : {len(prompts)}")
    print(f"  max_tokens  : {args.max_tokens}")
    print(f"  max_num_seqs: {args.max_num_seqs}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Engine init
    # ------------------------------------------------------------------
    print("\n[1/3] Initialising engine (model load + KV cache)...")
    t0 = time.perf_counter()
    config = VllmConfig(
        checkpoint_dir=CHECKPOINT_DIR,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=0.9,
    )
    engine = AsyncEngine(config, UniProcExecutor)
    t_init = time.perf_counter() - t0
    print(f"      done in {t_init:.1f}s  "
          f"(num_gpu_blocks={engine.num_gpu_blocks}, "
          f"max_seq_length={engine.max_seq_length})")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    print("\n[2/3] Loading tokenizer...")
    tokenizer = Tokenizer(CHECKPOINT_DIR)
    eos_id: int = tokenizer.eos_id
    print(f"      eos_id={eos_id}")

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    print("\n[3/3] Generating...")

    assert isinstance(engine.model_executor, UniProcExecutor)
    model_runner = engine.model_executor.model_runner
    max_seq_length = engine.max_seq_length

    formatted = [_build_llama3_prompt(p) if not args.no_chat_template else p
                 for p in prompts]
    token_id_lists = [tokenizer.encode(p, bos=False).tolist() for p in formatted]

    outputs: list[str] = []
    for i, (tids, orig) in enumerate(zip(token_id_lists, prompts)):
        print(f"\n  [{i}] {orig!r}  ({len(tids)} prompt tokens)")

        gen_toks, pfill_ms, tok_s = _generate_one(
            model_runner, tids, args.max_tokens, max_seq_length, eos_id
        )

        text = tokenizer.decode(torch.tensor(gen_toks)) if gen_toks else ""
        print(f"       prefill: {pfill_ms:.0f} ms | "
              f"decode: {tok_s:.1f} tok/s | "
              f"{len(gen_toks)} new tokens")
        outputs.append(text)

    print(f"\n{'='*60}")
    print("  Results")
    print(f"{'='*60}")
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"\n  Prompt [{i}]: {prompt!r}")
        print(f"  Output [{i}]: {output!r}")
    print(f"\n{'='*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
