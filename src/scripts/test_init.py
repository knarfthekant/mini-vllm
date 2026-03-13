import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pathlib import Path
from src.executor.uniproc_executor import UniProcExecutor
from src.config.vllm import VllmConfig


def test_imports_and_structure():
    """
    Verify that all simplified classes can be imported and that the class
    structure is as expected (without actually loading a real model checkpoint).
    """
    print("=== Verifying simplified model initiation structure ===")

    # Verify the simplified worker module has no leftover worker_base dependency
    from src.worker.gpu_worker import GPUWorker
    from src.worker.model_runner import GPUModelRunner

    # Confirm WorkerWrapper is gone and GPUWorker is plain class
    print(f"GPUWorker base classes: {[c.__name__ for c in GPUWorker.__bases__]}")
    assert GPUWorker.__bases__ == (object,), "GPUWorker should be a plain class now"

    # Confirm GPUModelRunner has a generate method
    assert hasattr(GPUModelRunner, "generate"), "GPUModelRunner should have a generate() method"

    # Confirm model_executor.loader no longer exists
    import importlib.util
    loader_spec = importlib.util.find_spec("src.model_executor.loader")
    assert loader_spec is None, "model_executor.loader should have been removed"

    print("GPUWorker and GPUModelRunner structure verified.")
    print("\nAll checks passed! Structure is simplified correctly.")
    print("Note: Actual model loading requires a real litgpt checkpoint_dir.")


if __name__ == "__main__":
    test_imports_and_structure()
