from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
from typing import Any

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "litgpt"))

from litgpt.attention_backends import DenseAttentionBackend, PagedAttentionBackend  # type: ignore[import-untyped]
from src.config.vllm import BLOCK_SIZE, PAGED_BLOCK_SIZE, VllmConfig
from src.request import Request
from src.worker.cache_manager import (
    KVCachePlan,
    PagedCacheManager,
    _build_paged_attention_metadata,
)
from src.worker.interface import SchedulerOutput


class _FakeAttn:
    def __init__(self) -> None:
        self.kv_cache = None
        self.backend = None

    def set_attention_backend(self, backend) -> None:
        self.backend = backend


class _FakeBlock:
    def __init__(self) -> None:
        self.attn = _FakeAttn()


def _make_runner(*, latent_attention=None, sliding_window_size=None):
    blocks = [_FakeBlock(), _FakeBlock()]
    config = SimpleNamespace(
        n_layer=len(blocks),
        n_query_groups=8,
        head_size=128,
        latent_attention=latent_attention,
        sliding_window_size=sliding_window_size,
        attention_logit_softcapping=None,
    )
    model = SimpleNamespace(
        config=config,
        max_seq_length=0,
        transformer=SimpleNamespace(
            wte=SimpleNamespace(weight=torch.zeros(1, dtype=torch.float16)),
            h=blocks,
        ),
    )
    return SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        vllm_config=VllmConfig(checkpoint_dir=Path("."), max_num_seqs=4),
    )


class PagedAttentionTest(unittest.TestCase):
    def test_metadata_prefill_without_prefix_cache(self) -> None:
        runner = _make_runner()
        request = Request([11, 12, 13])
        output = SchedulerOutput(
            requests=[request],
            input_ids=[[11, 12, 13]],
            positions=[[0, 1, 2]],
            block_tables=[[5]],
        )

        seq_lens = torch.tensor([3], dtype=torch.long)
        last_token_indices = torch.tensor([2], dtype=torch.long)
        metadata = _build_paged_attention_metadata(
            runner,  # type: ignore[arg-type]
            output,
            seq_lens,
            last_token_indices,
        )

        self.assertEqual(metadata.num_actual_tokens, 3)
        self.assertEqual(metadata.slot_mapping.tolist(), [80, 81, 82])
        self.assertEqual(metadata.block_tables.tolist(), [[5]])
        self.assertEqual(metadata.query_start_loc.tolist(), [0, 3])
        self.assertEqual(metadata.seq_lens.tolist(), [3])
        self.assertEqual(metadata.max_query_len, 3)
        self.assertEqual(metadata.max_seq_len, 3)
        self.assertEqual(metadata.last_token_indices.tolist(), [2])

    def test_metadata_decode_tracks_full_sequence_lengths(self) -> None:
        runner = _make_runner()
        request = Request(list(range(PAGED_BLOCK_SIZE)))
        request.num_cached_tokens = PAGED_BLOCK_SIZE
        request.append_token(999)
        output = SchedulerOutput(
            requests=[request],
            input_ids=[[999]],
            positions=[[PAGED_BLOCK_SIZE]],
            block_tables=[[2, 9]],
        )

        seq_lens = torch.tensor([1], dtype=torch.long)
        last_token_indices = torch.tensor([0], dtype=torch.long)
        metadata = _build_paged_attention_metadata(
            runner,  # type: ignore[arg-type]
            output,
            seq_lens,
            last_token_indices,
        )

        self.assertEqual(metadata.num_actual_tokens, 1)
        self.assertEqual(metadata.query_start_loc.tolist(), [0, 1])
        self.assertEqual(metadata.seq_lens.tolist(), [PAGED_BLOCK_SIZE + 1])
        self.assertEqual(metadata.slot_mapping.tolist(), [9 * PAGED_BLOCK_SIZE])
        self.assertEqual(metadata.block_tables.tolist(), [[2, 9]])

    def test_metadata_mixed_batch_builds_unified_query_offsets(self) -> None:
        runner = _make_runner()
        decode_request = Request(list(range(PAGED_BLOCK_SIZE)))
        decode_request.num_cached_tokens = PAGED_BLOCK_SIZE
        decode_request.append_token(777)
        prefill_request = Request([21, 22])

        output = SchedulerOutput(
            requests=[decode_request, prefill_request],
            input_ids=[[777], [21, 22]],
            positions=[[PAGED_BLOCK_SIZE], [0, 1]],
            block_tables=[[2, 9], [4]],
        )

        seq_lens = torch.tensor([1, 2], dtype=torch.long)
        last_token_indices = torch.tensor([0, 1], dtype=torch.long)
        metadata = _build_paged_attention_metadata(
            runner,  # type: ignore[arg-type]
            output,
            seq_lens,
            last_token_indices,
        )

        self.assertEqual(metadata.num_actual_tokens, 3)
        self.assertEqual(metadata.query_start_loc.tolist(), [0, 1, 3])
        self.assertEqual(metadata.seq_lens.tolist(), [PAGED_BLOCK_SIZE + 1, 2])
        self.assertEqual(metadata.max_query_len, 2)
        self.assertEqual(metadata.max_seq_len, PAGED_BLOCK_SIZE + 1)
        self.assertEqual(
            metadata.slot_mapping.tolist(),
            [9 * PAGED_BLOCK_SIZE, 4 * PAGED_BLOCK_SIZE, 4 * PAGED_BLOCK_SIZE + 1],
        )

    def test_initialize_kv_cache_allocates_block_major_layout_and_binds_backend(self) -> None:
        runner = _make_runner()
        manager = PagedCacheManager()
        plan = KVCachePlan(num_gpu_blocks=7, max_seq_length=PAGED_BLOCK_SIZE * 4, backend_name="paged")

        with mock.patch("src.worker.cache_manager.require_paged_attention_kernels"), mock.patch(
            "litgpt.attention_backends.require_paged_attention_kernels"
        ):
            state = manager.initialize_kv_cache(runner, plan)  # type: ignore[arg-type]

        self.assertEqual(state.layers[0].key_blocks.shape, (7, PAGED_BLOCK_SIZE, 8, 128))
        self.assertEqual(state.layers[0].value_blocks.shape, (7, PAGED_BLOCK_SIZE, 8, 128))
        self.assertIsInstance(runner.model.transformer.h[0].attn.backend, PagedAttentionBackend)
        self.assertIsNone(runner.model.transformer.h[0].attn.kv_cache)

    def test_paged_initialize_rejects_unsupported_model_features(self) -> None:
        runner = _make_runner(latent_attention={"enabled": True})
        manager = PagedCacheManager()
        plan = KVCachePlan(num_gpu_blocks=1, max_seq_length=PAGED_BLOCK_SIZE, backend_name="paged")

        with mock.patch("src.worker.cache_manager.require_paged_attention_kernels"), mock.patch(
            "litgpt.attention_backends.require_paged_attention_kernels"
        ):
            with self.assertRaisesRegex(RuntimeError, "latent attention"):
                manager.initialize_kv_cache(runner, plan)  # type: ignore[arg-type]

    def test_dense_backend_can_be_rebound_after_paged(self) -> None:
        from src.worker.cache_manager import _set_dense_attention_backends

        runner = _make_runner()
        _set_dense_attention_backends(runner)  # type: ignore[arg-type]

        self.assertIsInstance(runner.model.transformer.h[0].attn.backend, DenseAttentionBackend)

    def test_attention_backends_source_has_no_flash_attention_imports(self) -> None:
        import litgpt.attention_backends as attention_backends  # type: ignore[import-untyped]

        source = Path(attention_backends.__file__).read_text(encoding="utf-8")
        self.assertNotIn("flash_attn", source)
        self.assertNotIn("kernels-community/flash-attn2", source)

    def test_triton_paged_attention_source_has_no_flash_attention_imports(self) -> None:
        import litgpt.triton_paged_attention as triton_paged_attention  # type: ignore[import-untyped]

        source = Path(triton_paged_attention.__file__).read_text(encoding="utf-8")
        self.assertNotIn("flash_attn", source)
        self.assertNotIn("kernels-community/flash-attn2", source)


if __name__ == "__main__":
    unittest.main()
