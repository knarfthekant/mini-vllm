import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../litgpt"))

from pathlib import Path
from typing import Optional, Tuple

import torch
from lightning.fabric.utilities.load import _lazy_load as lazy_load

from litgpt.model import GPT
from litgpt.config import Config

from src.config.vllm import VllmConfig
from src.worker.model_input import ModelRunnerOutput, SchedulerOutput


class GPUModelRunner:
    """
    Manages loading and running the litgpt GPT model on a single GPU.

    vLLM-style execution pipeline
    ──────────────────────────────
    execute_model(scheduler_output)
        ├─ _update_states   – (stub) future KV-cache slot bookkeeping
        ├─ _prepare_inputs  – build (input_ids, positions, attn_metadata)
        └─ GPT.forward      – prefill forward pass → greedy sample
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self.model: Optional[GPT] = None
        self.device: Optional[torch.device] = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Instantiate GPT from config and load weights from checkpoint."""
        checkpoint_dir = Path(self.vllm_config.checkpoint_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = Config.from_file(checkpoint_dir / "model_config.yaml")

        self.model = GPT(config)
        self.model.eval()
        self.model.to(self.device)

        checkpoint_path = checkpoint_dir / "lit_model.pth"
        state_dict = lazy_load(checkpoint_path)
        state_dict = state_dict.get("model", state_dict)
        self.model.load_state_dict(state_dict, strict=True)

    def _update_states(self, scheduler_output: SchedulerOutput) -> None:
        """
        Stub: update internal request-to-KV-cache-slot mappings.

        In a full paged-attention implementation this method would consume the
        block tables from scheduler_output and maintain a slot-mapping tensor
        per live request.  Left empty until paged attention is added.
        """

    def _prepare_inputs(
        self, scheduler_output: SchedulerOutput
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """
        Build dense (B, T) input tensors from the scheduler output.

        Variable-length sequences are right-padded with 0s to the length of
        the longest sequence in the batch.  The attention-metadata slot is
        None for now; it will carry PagedAttention block tables once that
        layer is in place.

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

        attn_metadata = None  # placeholder for paged attention metadata

        return input_ids, positions, attn_metadata

    # ------------------------------------------------------------------
    # Main entry point called by the Executor
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """
        Run one inference step for the scheduled batch.

        Steps:
          1. _update_states  – sync KV cache bookkeeping (stub)
          2. _prepare_inputs – construct (B, T) input tensors
          3. GPT.forward     – full-attention prefill (no KV cache yet)
          4. greedy sample   – argmax over last-token logits

        Args:
            scheduler_output: batch of token sequences and their positions.

        Returns:
            ModelRunnerOutput with one sampled next-token per sequence.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        self._update_states(scheduler_output)

        input_ids, positions, _attn_metadata = self._prepare_inputs(scheduler_output)

        # input_pos is (B, T); litgpt accepts (T,) or (B, T) — pass as-is.
        # set_kv_cache is not called so the model runs in full-attention prefill
        # mode (no incremental caching).
        logits: torch.Tensor = self.model(input_ids, input_pos=positions)
        # logits shape: (B, T, vocab_size)

        next_token_ids = logits[:, -1, :].argmax(dim=-1).tolist()
        # next_token_ids: List[int] of length B

        return ModelRunnerOutput(sampled_token_ids=next_token_ids)
