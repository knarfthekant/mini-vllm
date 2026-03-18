from __future__ import annotations
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "litgpt"))

from pathlib import Path
from types import SimpleNamespace
import unittest

from src.config.vllm import BLOCK_SIZE, VllmConfig
from src.engine.allocator import (
    ClearDenseSlot,
    DenseSlotManager,
    MoveDenseSlot,
    PagedBlockManager,
)
from src.engine.scheduler import Scheduler
from src.request import Request
from src.worker.interface import ModelRunnerOutput


def _make_config(
    *,
    max_num_seqs: int = 4,
    max_num_batched_tokens: int = BLOCK_SIZE * 8,
) -> VllmConfig:
    return VllmConfig(
        checkpoint_dir=Path("."),
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )


def _make_scheduler(
    allocator: DenseSlotManager | PagedBlockManager,
    *,
    max_seq_length: int,
    max_num_seqs: int = 4,
    max_num_batched_tokens: int = BLOCK_SIZE * 8,
) -> Scheduler:
    return Scheduler(
        _make_config(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        allocator=allocator,
        max_seq_length=max_seq_length,
    )


def _start_two_dense_requests(*, max_seq_length: int = BLOCK_SIZE * 2) -> tuple[Scheduler, Request, Request]:
    scheduler = _make_scheduler(
        DenseSlotManager(max_num_seqs=2),
        max_seq_length=max_seq_length,
        max_num_seqs=2,
    )
    request1 = Request([11])
    request2 = Request([22])
    scheduler.add_request(request1)
    scheduler.add_request(request2)
    output = scheduler.schedule()
    scheduler.postprocess(output, ModelRunnerOutput(sampled_token_ids=[101, 202]))
    return scheduler, request1, request2


class AllocatorSchedulerUnificationTest(unittest.TestCase):
    def test_dense_free_emits_move_and_clear_and_updates_execution_key(self) -> None:
        allocator = DenseSlotManager(max_num_seqs=3)
        request1 = Request([1])
        request2 = Request([2])
        request3 = Request([3])

        for request in (request1, request2, request3):
            allocator.allocate(request)

        request2.num_cached_tokens = 7
        events = allocator.free(request2)

        self.assertEqual(
            events,
            [
                MoveDenseSlot(src_slot=2, dst_slot=1),
                ClearDenseSlot(slot=2),
            ],
        )
        self.assertEqual(request2.num_cached_tokens, 0)
        self.assertEqual(allocator.execution_key(request3), 1)
        self.assertEqual(allocator.export_batch_state(request3).block_table, [])

    def test_scheduler_orders_dense_running_requests_by_allocator_execution_key(self) -> None:
        scheduler, request1, request2 = _start_two_dense_requests()

        scheduler.running = [request2, request1]
        output = scheduler.schedule()

        self.assertEqual(output.requests, [request1, request2])
        self.assertEqual(output.block_tables, [[], []])

    def test_abort_and_length_cap_share_dense_release_events(self) -> None:
        abort_scheduler, abort_request, _ = _start_two_dense_requests(max_seq_length=8)
        abort_result = abort_scheduler.abort_request(abort_request.request_id)
        self.assertIsNotNone(abort_result)

        length_cap_scheduler, capped_request, surviving_request = _start_two_dense_requests(
            max_seq_length=2
        )
        capped_request.num_tokens = length_cap_scheduler.max_seq_length + 1
        schedule_output = length_cap_scheduler.schedule()
        pending_result = length_cap_scheduler.drain_pending_result()

        expected_events = [
            MoveDenseSlot(src_slot=1, dst_slot=0),
            ClearDenseSlot(slot=1),
        ]
        assert abort_result is not None
        self.assertEqual(abort_result.allocator_events, expected_events)
        self.assertEqual(pending_result.allocator_events, expected_events)
        self.assertEqual(pending_result.finished_requests, [capped_request])
        self.assertEqual(schedule_output.requests, [surviving_request])

    def test_paged_scheduler_uses_exported_block_tables(self) -> None:
        allocator = PagedBlockManager(num_gpu_blocks=16)
        scheduler = _make_scheduler(
            allocator,
            max_seq_length=BLOCK_SIZE * 4,
            max_num_batched_tokens=BLOCK_SIZE * 4,
        )
        request = Request(list(range(BLOCK_SIZE + 3)))

        scheduler.add_request(request)
        output = scheduler.schedule()

        self.assertFalse(hasattr(request, "block_table"))
        self.assertEqual(output.requests, [request])
        self.assertEqual(
            output.block_tables,
            [allocator.export_batch_state(request).block_table],
        )

    def test_paged_decode_crosses_block_boundary_without_allocator_events(self) -> None:
        allocator = PagedBlockManager(num_gpu_blocks=16)
        scheduler = _make_scheduler(
            allocator,
            max_seq_length=BLOCK_SIZE * 4,
            max_num_batched_tokens=BLOCK_SIZE * 4,
        )
        request = Request(list(range(BLOCK_SIZE)))

        scheduler.add_request(request)
        prefill_output = scheduler.schedule()
        prefill_result = scheduler.postprocess(
            prefill_output,
            ModelRunnerOutput(sampled_token_ids=[999]),
        )
        decode_output = scheduler.schedule()
        decode_result = scheduler.postprocess(
            decode_output,
            ModelRunnerOutput(sampled_token_ids=[1000]),
        )

        self.assertEqual(prefill_result.allocator_events, [])
        self.assertEqual(len(prefill_output.block_tables[0]), 1)
        self.assertEqual(decode_output.requests, [request])
        self.assertEqual(len(decode_output.block_tables[0]), 2)
        self.assertEqual(decode_result.allocator_events, [])

    def test_paged_free_resets_request_cache_state(self) -> None:
        allocator = PagedBlockManager(num_gpu_blocks=16)
        scheduler = _make_scheduler(
            allocator,
            max_seq_length=BLOCK_SIZE * 4,
            max_num_batched_tokens=BLOCK_SIZE * 4,
        )
        request = Request(list(range(BLOCK_SIZE)))

        scheduler.add_request(request)
        output = scheduler.schedule()
        scheduler.postprocess(output, ModelRunnerOutput(sampled_token_ids=[1234]))

        events = allocator.free(request)

        self.assertEqual(events, [])
        self.assertEqual(request.num_cached_tokens, 0)
        with self.assertRaises(KeyError):
            allocator.export_batch_state(request)

    def test_standard_cache_manager_applies_dense_allocator_events(self) -> None:
        try:
            import torch
            from src.worker.cache_manager import StandardCacheManager
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch-dependent cache manager test skipped: {exc}")

        manager = StandardCacheManager()

        def make_layer(fill_value: float) -> SimpleNamespace:
            key = torch.full((2, 3, 2), fill_value)
            value = torch.full((2, 3, 2), fill_value * 10)
            kv_cache = SimpleNamespace(k=key.clone(), v=value.clone())
            kv_cache.k[0].zero_()
            kv_cache.v[0].zero_()
            return SimpleNamespace(attn=SimpleNamespace(kv_cache=kv_cache))

        runner = SimpleNamespace(
            model=SimpleNamespace(
                transformer=SimpleNamespace(h=[make_layer(1.0), make_layer(2.0)])
            )
        )

        manager.apply_allocator_events(
            runner,
            [
                MoveDenseSlot(src_slot=1, dst_slot=0),
                ClearDenseSlot(slot=1),
            ],
        )

        for expected_fill, block in zip((1.0, 2.0), runner.model.transformer.h):
            kv = block.attn.kv_cache
            self.assertTrue(torch.all(kv.k[0] == expected_fill))
            self.assertTrue(torch.all(kv.v[0] == expected_fill * 10))
            self.assertEqual(torch.count_nonzero(kv.k[1]).item(), 0)
            self.assertEqual(torch.count_nonzero(kv.v[1]).item(), 0)


if __name__ == "__main__":
    unittest.main()
