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

logging.basicConfig(
    level=logging.INFO,  # change to DEBUG if needed
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

def run(name: str, fn):
    """Run a single test, record result, never abort the suite."""
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception:
        print(f"  FAIL  {name}")
        traceback.print_exc()

def test_engine_init():
    tokenizer = Tokenizer(CHECKPOINT_DIR)
    eos_id = tokenizer.eos_id # type: ignore[assignment]
    engine = AsyncEngine(VLLM_CONFIG, eos_token_id=eos_id)
    print(f"  Engine initialized with {engine.num_gpu_blocks} GPU blocks and {engine.max_seq_length} sequence length")
    print(f"  EOS ID: {eos_id}")

run("test_engine_init", test_engine_init)