import logging
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../litgpt"))

from pathlib import Path
from typing import Optional, Tuple

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

        Memory-efficient strategy to avoid OOM on 16 GB GPUs:

          1. Build the model on the ``meta`` device — zero bytes, just structure.
          2. Open the checkpoint with ``mmap=True`` so only the pages currently
             being read are faulted into RAM rather than the full 16 GB at once.
          3. ``to_empty(device)`` allocates uninitialised tensors on the GPU.
          4. ``load_state_dict(assign=True)`` moves weights directly from the
             memory-mapped CPU buffer onto the GPU, preserving the checkpoint
             dtype (bfloat16).  No intermediate fp32 copy is created.

        Without this approach the naïve flow (GPT(config) → .to(device)) creates
        a full fp32 copy on CPU (~30 GB) before the GPU move, which OOMs on
        machines where GPU VRAM ≈ model size.
        """
        checkpoint_dir = Path(self.vllm_config.checkpoint_dir)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = Config.from_file(checkpoint_dir / "model_config.yaml")

        # Step 1 — model structure on meta device: 0 bytes allocated
        with torch.device("meta"):
            self._model = GPT(config)
        self.model.eval()

        # Step 2 — allocate empty (uninitialised) tensors on the target device.
        #           Parameters land on GPU but hold garbage — filled in step 4.
        self.model.to_empty(device=self.device)

        # Step 3 — memory-map the .pth file; pages are faulted in on demand
        checkpoint_path = checkpoint_dir / "lit_model.pth"
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,           # stream weights from disk rather than loading all at once
            weights_only=False,  # litgpt uses custom storage objects
        )
        state_dict = state_dict.get("model", state_dict)

        # Step 4 — copy weights from the mmap CPU buffer into the GPU tensors.
        #           Without assign=True, load_state_dict copies each tensor from
        #           the CPU mmap into the existing GPU parameters from step 2.
        #           (assign=True would *replace* them with CPU tensors, leaving
        #           the model on CPU — which is the bug we are fixing.)
        self.model.load_state_dict(state_dict, strict=True)

        dtype = next(self.model.parameters()).dtype
        logger.info("Model loaded on %s in %s", self.device, dtype)

    # ------------------------------------------------------------------
    # KV-cache initialization  (mirrors vLLM v1 Worker interface)
    # ------------------------------------------------------------------

    def profile_run(self) -> None:
        """
        Run a single dummy forward pass to warm up CUDA kernels and stabilise
        GPU memory allocations before reading memory stats.

        Uses batch_size=1 and BLOCK_SIZE tokens to keep the profile cheap
        while still triggering all kernel paths.
        """
        dummy = torch.zeros((1, BLOCK_SIZE), dtype=torch.long, device=self.device)
        with torch.inference_mode():
            self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    def determine_available_memory(self) -> int:
        """
        Profile the model and return the number of bytes available for KV
        cache allocation.

        Follows nano-vllm's approach:
          available = total * gpu_memory_utilization
                      - used          (current physical allocation)
                      - peak          (peak PyTorch allocation during forward)
                      + current       (PyTorch current allocation; freed after forward)

        The ``peak - current`` delta captures activation memory at the worst
        point of a forward pass.  On CPU, a conservative fixed budget is
        returned so the logic path remains exercisable without a GPU.
        """
        self.profile_run()

        if self.device.type != "cuda":
            # CPU fallback: allocate enough for the full context window.
            return _CPU_FALLBACK_MEMORY_BYTES

        free, total = torch.cuda.mem_get_info()
        used = total - free
        stats = torch.cuda.memory_stats()
        peak = stats["allocated_bytes.all.peak"]
        current = stats["allocated_bytes.all.current"]

        available = int(
            total * self.vllm_config.gpu_memory_utilization
            - used
            - peak
            + current
        )
        available = max(0, available)
        logger.info(
            "Memory profiling: total=%.1f GiB, used=%.1f GiB, "
            "peak=%.1f GiB, available_for_kv=%.1f GiB",
            total / _GiB,
            used / _GiB,
            peak / _GiB,
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
        dtype_bytes: int = self.model.transformer.wte.weight.element_size()
        max_num_seqs: int = self.vllm_config.max_num_seqs

        # rope_cache_length: number of rotary elements in K's last dimension.
        rope_cache_length: int = self.model.rope_cache_length()
        # K last-dim width (see litgpt build_kv_cache)
        k_head_dim: int = rope_cache_length + cfg.head_size - cfg.rope_n_elem

        # Bytes consumed per token position across the whole batch:
        #   2 sides (K/V) × layers × batch × query-groups × head-dim
        bytes_per_token: int = (
            2 * cfg.n_layer * max_num_seqs * cfg.n_query_groups
            * k_head_dim * dtype_bytes
        )

        if bytes_per_token == 0:
            raise RuntimeError("bytes_per_token is zero; model config may be invalid")

        max_seq_length: int = available_bytes // bytes_per_token
        # Respect the model's own context window
        max_seq_length = min(max_seq_length, cfg.block_size)
        # Round down to BLOCK_SIZE boundary (≥ 1 block)
        max_seq_length = max(BLOCK_SIZE, (max_seq_length // BLOCK_SIZE) * BLOCK_SIZE)

        num_gpu_blocks: int = max_seq_length // BLOCK_SIZE

        logger.info(
            "KV cache sizing: %d blocks × %d tokens/block = %d max_seq_length "
            "(batch_size=%d, %.1f GiB)",
            num_gpu_blocks,
            BLOCK_SIZE,
            max_seq_length,
            max_num_seqs,
            num_gpu_blocks * BLOCK_SIZE * bytes_per_token / max_seq_length / _GiB,
        )
        return num_gpu_blocks, max_seq_length

    def initialize_kv_cache(self, num_gpu_blocks: int, max_seq_length: int) -> None:
        """
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
