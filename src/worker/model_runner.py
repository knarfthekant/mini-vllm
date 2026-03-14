import logging
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../litgpt"))

from pathlib import Path
from typing import Tuple

import torch

from litgpt.model import GPT  # type: ignore[import-untyped]
from litgpt.config import Config  # type: ignore[import-untyped]

from src.config.vllm import BLOCK_SIZE, VllmConfig
from src.worker.model_input import ModelRunnerOutput, SchedulerOutput

logger = logging.getLogger(__name__)


class GPUModelRunner:
    """
    Manages loading and running the litgpt GPT model on a single GPU.

    Wraps litgpt.model.GPT directly so the attention layer can later be
    replaced with a paged-attention variant without touching the rest of the
    engine stack.

    Lifecycle
    ─────────
    1. load_model()                 – instantiate GPT and load weights
    2. profile_run()                – warm-up forward for memory profiling
    3. determine_available_memory() – compute bytes free for KV cache
    4. compute_num_gpu_blocks()     – translate bytes → (num_blocks, max_seq_len)
    5. initialize_kv_cache()        – call GPT.set_kv_cache() to allocate buffers

    vLLM-style execution pipeline (per step)
    ─────────────────────────────────────────
    execute_model(scheduler_output)
        ├─ _update_states   – (stub) future KV-cache slot bookkeeping
        ├─ _prepare_inputs  – build (input_ids, positions, attn_metadata)
        └─ GPT.forward      – forward pass → greedy sample
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self._model: GPT | None = None
        self._device: torch.device | None = None
        # Free GPU bytes recorded immediately before model weights are loaded.
        # Used by determine_available_memory() to measure the real memory delta
        # caused by loading the model and running a warm-up forward pass.
        self._init_gpu_free_bytes: int = 0

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

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Instantiate GPT from checkpoint config and load weights.

        Loading strategy — minimal CPU RAM, correct dtype, single VRAM copy:

          1. ``torch.load(map_location=device)`` streams each tensor directly
             from disk → GPU through a small I/O buffer.  CPU RAM stays near
             zero throughout (one tensor worth of read buffer at a time).
             The previous mmap=True + map_location="cpu" approach faulted all
             ~15 GiB of pages into RAM during load_state_dict, OOM-ing WSL2.

          2. The checkpoint dtype (bfloat16) is read from the first tensor so
             the meta model can be cast before assign.  Without this the meta
             model defaults to float32, doubling VRAM usage (~15 GiB → ~30 GiB)
             and leaving no room for the KV cache.

          3. ``load_state_dict(assign=True)`` wires the already-GPU tensors
             directly into model parameters — no second VRAM copy, and no
             ``to_empty()`` pre-allocation is needed.
        """
        checkpoint_dir = Path(self.vllm_config.checkpoint_dir)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Snapshot free GPU memory before any model tensors are allocated.
        # determine_available_memory() computes the delta against this value to
        # find how much memory the model (+ warm-up activations) consumed.
        if self._device.type == "cuda":
            self._init_gpu_free_bytes = torch.cuda.mem_get_info()[0]

        config = Config.from_file(checkpoint_dir / "model_config.yaml")
        checkpoint_path = checkpoint_dir / "lit_model.pth"

        # Step 1 — stream weights from disk directly onto the target device.
        #           CPU RAM usage stays near zero: PyTorch reads one tensor at
        #           a time through an internal I/O buffer, then frees it.
        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,  # litgpt checkpoints may contain metadata objects
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

    # ------------------------------------------------------------------
    # KV-cache initialization  (mirrors vLLM v1 Worker interface)
    # ------------------------------------------------------------------

    def profile_run(self) -> None:
        """
        Run a single dummy forward pass to warm up CUDA kernels and let all
        persistent allocations (model weights, CUDA graphs, etc.) settle before
        the memory snapshot in ``determine_available_memory`` is taken.

        Uses batch_size=1 and BLOCK_SIZE tokens to keep the profile cheap
        while still triggering all kernel paths.
        """
        dummy = torch.zeros((1, BLOCK_SIZE), dtype=torch.long, device=self.device)
        with torch.inference_mode():
            self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def determine_available_memory(self) -> int:
        """
        Return the number of bytes available for KV cache allocation.

        Follows the same approach as vLLM and nano-vllm:

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

    def compute_num_gpu_blocks(self, available_bytes: int) -> Tuple[int, int]:
        """
        Translate available bytes into ``(num_gpu_blocks, max_seq_length)``.

        KV-cache memory footprint for litgpt's static buffers:

            bytes = 2           (K + V)
                  × n_layer
                  × max_num_seqs
                  × n_query_groups
                  × max_seq_length
                  × k_head_dim   (= rope_cache_length + head_size - rope_n_elem)
                  × dtype_bytes

        We solve for ``max_seq_length``, cap at the model's context window,
        round down to ``BLOCK_SIZE``, and derive ``num_blocks`` from that.

        The ``k_head_dim`` formula comes from litgpt's ``build_kv_cache``:
        the stored K tensor includes the non-rotary residual dimensions, so
        its last axis is slightly wider than ``head_size`` when
        ``rotary_percentage < 1``.  When ``rotary_percentage == 1`` it equals
        ``head_size``.  We use the actual rope_cache_length from the model's
        registered buffer for accuracy.

        Args:
            available_bytes: Bytes available for KV-cache tensors.

        Returns:
            num_gpu_blocks: Number of BLOCK_SIZE-token blocks in the KV pool.
            max_seq_length: Maximum per-sequence context the cache supports.
        """
        cfg = self.model.config
        dtype_bytes: int = self.model.transformer.wte.weight.element_size() # type: ignore[attr-defined]
        max_num_seqs: int = self.vllm_config.max_num_seqs

        # rope_cache_length: number of rotary elements in K's last dimension.
        rope_cache_length: int = self.model.rope_cache_length() # type: ignore[attr-defined]
        # K last-dim width (see litgpt build_kv_cache)
        assert cfg.head_size is not None, "Config.head_size must be set after __post_init__"
        assert cfg.n_query_groups is not None, "Config.n_query_groups must be set after __post_init__"
        rope_n_elem: int = int(cfg.rope_n_elem) # type: ignore[attr-defined]
        k_head_dim: int = rope_cache_length + cfg.head_size - rope_n_elem

        # Bytes consumed per token position across the whole batch:
        #   2 sides (K/V) × layers × batch × query-groups × head-dim
        bytes_per_token: int = (
            2 * cfg.n_layer * max_num_seqs * cfg.n_query_groups
            * k_head_dim * dtype_bytes
        )

        if bytes_per_token == 0:
            raise RuntimeError("bytes_per_token is zero; model config may be invalid")

        # Reserve a fraction of available bytes for activation memory during
        # prefill (nano-vllm / vLLM style).  Using 100% for KV leaves no headroom
        # and causes OOM or extreme slowness when allocating 50K+ token caches.
        kv_cache_fraction: float = 0.9
        bytes_for_kv: int = int(available_bytes * kv_cache_fraction)

        # This is the maximum sequence length that can be supported by the available memory.
        max_seq_length: int = bytes_for_kv // bytes_per_token

        # Cap at user-configurable max (keeps init and prefill fast)
        if self.vllm_config.max_model_len is not None:
            max_seq_length = min(max_seq_length, self.vllm_config.max_model_len)

        # Respect the model's own context window
        max_seq_length = min(max_seq_length, cfg.block_size)
        # Round down to BLOCK_SIZE boundary (≥ 1 block)
        max_seq_length = max(BLOCK_SIZE, (max_seq_length // BLOCK_SIZE) * BLOCK_SIZE)

        num_gpu_blocks: int = max_seq_length // BLOCK_SIZE

        kv_cache_gib = (max_seq_length * bytes_per_token) / _GiB
        logger.info(
            "KV cache sizing: %d blocks × %d tokens/block = %d max_seq_length "
            "(batch_size=%d, %.2f GiB)",
            num_gpu_blocks,
            BLOCK_SIZE,
            max_seq_length,
            max_num_seqs,
            kv_cache_gib,
        )
        return num_gpu_blocks, max_seq_length

    def initialize_kv_cache(self, num_gpu_blocks: int, max_seq_length: int) -> None:
        """
        TODO: Implement paged attention support in the future.
        Allocate litgpt's static KV-cache buffers and bind them to the model.

        Calls ``GPT.set_kv_cache(batch_size, max_seq_length)`` which
        allocates one ``KVCache`` (two registered ``torch.Tensor`` buffers for
        K and V) per transformer block.  Also updates ``model.max_seq_length``
        so the RoPE cache is resized to match.

        These buffers are not yet used during ``execute_model``; that requires
        the scheduler to pass correct positions for decode steps.  The cache
        will be populated automatically by litgpt's ``KVCache.forward`` once
        ``input_pos`` is passed in the forward call.

        Args:
            num_gpu_blocks: Number of BLOCK_SIZE-token blocks.
            max_seq_length: Maximum sequence length the cache supports.
        """
        max_num_seqs: int = self.vllm_config.max_num_seqs

        # Resize RoPE buffers first; set_kv_cache then sees the correct size.
        self.model.max_seq_length = max_seq_length

        self.model.set_kv_cache(
            batch_size=max_num_seqs,
            max_seq_length=max_seq_length,
            device=self.device,
        )
        logger.info(
            "KV cache allocated: batch_size=%d, max_seq_length=%d (%d blocks)",
            max_num_seqs,
            max_seq_length,
            num_gpu_blocks,
        )

    # ------------------------------------------------------------------
    # vLLM-style execution pipeline
    # ------------------------------------------------------------------

    def _update_states(self, scheduler_output: SchedulerOutput) -> None:
        """
        Stub: update request-to-KV-cache-slot mappings.

        A full paged-attention implementation would consume block tables from
        the scheduler output here.  Left empty until that layer is added.
        """

    def _prepare_inputs(
        self, scheduler_output: SchedulerOutput
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """
        Build dense ``(B, T)`` input tensors from the scheduler output.

        Variable-length sequences are right-padded with 0s to the length of
        the longest sequence in the batch.  The ``attn_metadata`` slot is
        ``None`` for now; it will carry paged-attention block tables later.

        Returns:
            input_ids:     LongTensor  (B, T)
            positions:     LongTensor  (B, T)
            attn_metadata: None  (stub)
        """
        seqs = scheduler_output.input_ids
        pos_seqs = scheduler_output.positions

        max_len = max(len(s) for s in seqs)

        padded_ids = [s + [0] * (max_len - len(s)) for s in seqs]
        padded_pos = [p + [0] * (max_len - len(p)) for p in pos_seqs]

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(padded_pos, dtype=torch.long, device=self.device)

        return input_ids, positions, None  # attn_metadata is stub

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
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        self._update_states(scheduler_output)
        input_ids, positions, _attn_metadata = self._prepare_inputs(scheduler_output)

        logits: torch.Tensor = self.model(input_ids, input_pos=positions)
        # logits: (B, T, vocab_size) — take last token for each sequence
        next_token_ids = logits[:, -1, :].argmax(dim=-1).tolist()

        return ModelRunnerOutput(sampled_token_ids=next_token_ids)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_GiB: int = 1024 ** 3

# On CPU there is no GPU memory to query.  We return enough bytes to cover
# the model's full context window at typical model sizes, so the CPU path
# exercises the same code as the GPU path.
_CPU_FALLBACK_MEMORY_BYTES: int = 16 * _GiB
