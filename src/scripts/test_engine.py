import asyncio
from pathlib import Path
import sys
import os
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "litgpt"))

from src.config.vllm import BLOCK_SIZE, VllmConfig
from src.engine.async_engine import AsyncEngine
from src.worker.interface import SchedulerOutput

from litgpt.tokenizer import Tokenizer  # type: ignore[import-untyped]

import logging
import torch

logging.basicConfig(
    level=logging.DEBUG,  # change to DEBUG if needed
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
CHECKPOINT_DIR = Path(ROOT) / "checkpoints/meta-llama/Meta-Llama-3.1-8B-Instruct"

VLLM_CONFIG = VllmConfig(
    checkpoint_dir=CHECKPOINT_DIR,
    max_num_seqs=2,
    gpu_memory_utilization=0.9,
)

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = AsyncEngine(VLLM_CONFIG)
    return _engine


def run(name: str, fn):
    """Run a single test, record result, never abort the suite."""
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception:
        print(f"  FAIL  {name}")
        traceback.print_exc()


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_engine_init():
    engine = get_engine()
    print(f"  Engine initialized with {engine.num_gpu_blocks} GPU blocks and {engine.max_seq_length} sequence length")
    print(f"  EOS ID: {engine.eos_token_id}")

def test_add_request_str():
    engine = get_engine()
    request = engine.add_request("Hello, world or how are you?")
    print(f"  Request added: {request.request_id}")
    get_result = engine.get_request(request.request_id)
    print(f"  Request: {get_result}")
    abort_result = engine.abort_request(request.request_id)
    print(f"  Abort result: {abort_result}")

async def test_engine_step():
    engine = get_engine()
    request = engine.add_request("Explain kv cache")
    request = engine.add_request("Explain tensor parallelism")
    print(f"  Request added: {request.request_id}")
    result = await engine.run_until_idle()
    print(f"  Result0: {engine.tokenizer.decode(torch.tensor(result[0].token_ids))}")
    print(f"  Result1: {engine.tokenizer.decode(torch.tensor(result[1].token_ids))}")


async def test_engine_step_delayed_second_request():
    """Run first request for a while, then add the second request and run until both complete."""
    engine = get_engine()
    req1 = engine.add_request("Explain kv cache")
    print(f"  First request added: {req1.request_id}")

    # Run the engine for a while so the first request generates some tokens
    steps = 0
    max_steps_before_second = 40
    while engine.has_unfinished_requests() and steps < max_steps_before_second:
        engine.step()
        steps += 1
        await asyncio.sleep(0)
    r1_after_warmup = engine.get_request(req1.request_id)
    assert r1_after_warmup is not None
    print(f"  After {steps} steps, first request has {r1_after_warmup.num_completion_tokens} completion tokens")

    req2 = engine.add_request("Explain tensor parallelism")
    print(f"  Second request added: {req2.request_id}")

    result = await engine.run_until_idle()
    by_id = {r.request_id: r for r in result}
    r1 = by_id[req1.request_id]
    r2 = by_id[req2.request_id]
    print(f"  Result1 (first):  {engine.tokenizer.decode(torch.tensor(r1.token_ids))[:200]}...")
    print(f"  Result2 (second): {engine.tokenizer.decode(torch.tensor(r2.token_ids))[:200]}...")


# run("test_engine_init", test_engine_init)
# run("test_add_request_str", test_add_request_str)
# run("test_engine_step", lambda: asyncio.run(test_engine_step()))
run("test_engine_step_delayed_second_request", lambda: asyncio.run(test_engine_step_delayed_second_request()))