import logging
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../litgpt"))

from pathlib import Path
from typing import Tuple

import torch

from litgpt.model import GPT  # type: ignore[import-untyped]

from src.config.vllm import BLOCK_SIZE, VllmConfig
from src.worker.cache_manager import (
    BaseCacheManager,
    DenseKVCacheState,
    KVCachePlan,
    ModelExecutionInputs,
    PagedKVCacheState,
    build_cache_manager,
)
from src.worker.interface import SchedulerOutput, ModelRunnerOutput

logger = logging.getLogger(__name__)

class ModelRunner:
    """
    Manages loading and running the litgpt GPT model on a single GPU.

    The runner owns the model lifecycle and delegates attention-specific KV
    cache behavior to a pluggable Strategy. The standard backend allocates
    dense per-sequence KV tensors; the paged backend allocates a shared block
    pool for future paged-attention execution.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        logger.info("Initializing ModelRunner")
        self.vllm_config = vllm_config
        self._cache_manager: BaseCacheManager = build_cache_manager(vllm_config.kv_cache_manager)
        self._model: GPT | None = None
        self._device: torch.device | None = None
        self._init_gpu_free_bytes = 0
        self._kv_cache_plan: KVCachePlan | None = None
        self._kv_cache_state: DenseKVCacheState | PagedKVCacheState | None = None

    @property
    def model(self) -> GPT:
        if self._model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        return self._model

    @property
    def device(self) -> torch.device:
        if self._device is None:
            raise RuntimeError("Device is not set. Call load_model() first.")
        return self._device

    @property
    def cache_manager_name(self) -> str:
        return self._cache_manager.name

    @property
    def kv_cache_plan(self) -> KVCachePlan:
        if self._kv_cache_plan is None:
            raise RuntimeError("KV cache is not initialized. Call initialize_kv_cache() first.")
        return self._kv_cache_plan

    @property
    def kv_cache_state(self) -> DenseKVCacheState | PagedKVCacheState | None:
        return self._kv_cache_state

    def load_model(self) -> None:
        """
        Instantiate GPT from checkpoint config and load weights.
        """

        checkpoint_dir = Path(self.vllm_config.checkpoint_dir)

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            logger.warning("No CUDA available, using CPU")
            self._device = torch.device("cpu")

        # Snapshot free GPU memory before any model tensors are allocated.
        # used by determine_available_memory()
        if self.device.type == "cuda":
            self._init_gpu_free_bytes = torch.cuda.mem_get_info()[0]

        config = self.vllm_config.model_config
        checkpoint_path = checkpoint_dir / "lit_model.pth"

        # Step 1 — stream weights from disk directly onto the target device.
        #           CPU RAM usage stays near zero: PyTorch reads one tensor at
        #           a time through an internal I/O buffer, then frees it.
        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        state_dict = state_dict.get("model", state_dict)

        # Step 2 — build model structure on the meta device (0 bytes), then
        #           cast to the checkpoint dtype so that assign in step 3 does
        #           not silently upcast bf16 weights to fp32.
        checkpoint_dtype = next(iter(state_dict.values())).dtype
        with torch.device("meta"):
            self._model = GPT(config)
        self.model.to(dtype=checkpoint_dtype)
        self.model.eval()

        # Step 3 — assign the already-loaded GPU tensors to the model.
        #           assign=True replaces each meta parameter in-place; no second
        #           VRAM allocation and no to_empty() pre-allocation needed.
        self.model.load_state_dict(state_dict, strict=True, assign=True)

        # Step 4 — rebuild non-persistent RoPE buffers (cos/sin) on the real
        #           device.  GPT.__init__ calls the max_seq_length setter while
        #           inside `with torch.device("meta")`, registering cos/sin as
        #           meta-device tensors.  They are persistent=False so they are
        #           not saved in the checkpoint and assign=True never replaces
        #           them.  Without this step, forward() crashes with:
        #             "Tensor on device meta is not on the expected device cuda"
        cos, sin = self.model.rope_cache(device=self.device)
        self.model.register_buffer("cos", cos, persistent=False)
        self.model.register_buffer("sin", sin, persistent=False)

        dtype = next(self.model.parameters()).dtype
        logger.info("Model loaded on %s in %s", self.device, dtype)


    def profile_run(self) -> None:
        """
        Run a single dummy forward pass to warm up CUDA kernels and let all
        persistent allocations (model weights, CUDA graphs, etc.) settle before
        the memory snapshot in ``determine_available_memory`` is taken.
        """
        dummy = torch.zeros((1, BLOCK_SIZE), dtype=torch.long, device=self.device)
        with torch.inference_mode():
            self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def determine_available_memory(self) -> int:
        """
        Return the number of bytes available for KV cache allocation.

          1. ``_init_gpu_free_bytes`` is recorded in ``load_model()`` *before*
             any model tensors are placed on the GPU.
          2. A warm-up forward pass (``profile_run``) is executed so that all
             persistent CUDA allocations have been made.
          3. ``torch.cuda.empty_cache()`` returns unused cached blocks to the
             CUDA driver so that ``mem_get_info`` reflects real free memory.
          4. ``peak_used = _init_gpu_free_bytes - current_free`` is the physical
             memory consumed by the model weights, CUDA context, and the worst-
             case activation footprint of a single forward pass.
          5. ``available = total * gpu_memory_utilization - peak_used``

        This approach relies only on the CUDA driver's view of free/total memory
        and avoids PyTorch's caching-allocator internals (``memory_stats`` peak
        vs current), which are unreliable as an "available for KV cache" proxy.
        """
        self.profile_run()

        if self.device.type != "cuda":
            return _CPU_FALLBACK_MEMORY_BYTES

        # Release PyTorch's cached-but-idle blocks so the CUDA driver reports
        # them as free in the mem_get_info call below.
        torch.cuda.empty_cache()

        free, total = torch.cuda.mem_get_info()

        # Memory physically consumed since before load_model() was called.
        # Includes: model weights + CUDA context + warm-up activation residue.
        peak_used = self._init_gpu_free_bytes - free

        available = int(total * self.vllm_config.gpu_memory_utilization - peak_used)
        available = max(0, available)

        logger.info(
            "Memory profiling: total=%.1f GiB, before_load_free=%.1f GiB, "
            "now_free=%.1f GiB, peak_used=%.1f GiB, available_for_kv=%.1f GiB",
            total / _GiB,
            self._init_gpu_free_bytes / _GiB,
            free / _GiB,
            peak_used / _GiB,
            available / _GiB,
        )
        return available

    def plan_kv_cache(self, available_bytes: int | None = None) -> KVCachePlan:
        if available_bytes is None:
            available_bytes = self.determine_available_memory()
        return self._cache_manager.build_kv_cache_plan(self, available_bytes)

    def compute_num_gpu_blocks(self, available_bytes: int) -> Tuple[int, int]:
        plan = self.plan_kv_cache(available_bytes)
        return plan.num_gpu_blocks, plan.max_seq_length

    def initialize_kv_cache(self, plan: KVCachePlan | None = None) -> KVCachePlan:
        if plan is None:
            plan = self.plan_kv_cache()
        self._kv_cache_state = self._cache_manager.initialize_kv_cache(self, plan)
        self._kv_cache_plan = plan
        return plan

    # ------------------------------------------------------------------
    # Execution pipeline
    # ------------------------------------------------------------------

    def _update_states(self, scheduler_output: SchedulerOutput) -> None:
        self._cache_manager.update_states(self, scheduler_output)

    def _prepare_inputs(self, scheduler_output: SchedulerOutput) -> ModelExecutionInputs:
        return self._cache_manager.prepare_model_inputs(self, scheduler_output)

    def move_sequence_cache(self, src_slot: int, dst_slot: int) -> None:
        # allow tensor to be modified in-place
        with torch.inference_mode():
            self._cache_manager.move_sequence_cache(self, src_slot, dst_slot)

    def clear_sequence_cache(self, slot: int) -> None:
        # allow tensor to be modified in-place
        with torch.inference_mode():
            self._cache_manager.clear_sequence_cache(self, slot)

    @torch.inference_mode()
    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """
        Run one inference step for the scheduled batch.

        Steps:
          1. _update_states  – sync KV cache bookkeeping (stub)
          2. _prepare_inputs – construct (B, T) input tensors
          3. GPT.forward     – forward pass → greedy argmax

        Args:
            scheduler_output: batch of token sequences and their positions.

        Returns:
            ModelRunnerOutput with one sampled next-token per sequence.
        """
        if self._model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        self._update_states(scheduler_output)
        model_inputs = self._prepare_inputs(scheduler_output)

        logits = self._cache_manager.forward(self, model_inputs)
        # logits: (B, T, vocab_size) — take last token for each sequence
        next_token_ids = logits[:, -1, :].argmax(dim=-1).tolist()

        return ModelRunnerOutput(sampled_token_ids=next_token_ids)


GPUModelRunner = ModelRunner


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_GiB: int = 1024 ** 3

# On CPU there is no GPU memory to query.  We return enough bytes to cover
# the model's full context window at typical model sizes, so the CPU path
# exercises the same code as the GPU path.
_CPU_FALLBACK_MEMORY_BYTES: int = 16 * _GiB
