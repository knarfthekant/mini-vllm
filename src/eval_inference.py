#!/usr/bin/env python3
"""
Inference benchmark harness for mini-vLLM.

This script focuses on request-level latency and throughput measurements for
the current dense backend while keeping the same workload and report format for
future dense-vs-paged comparisons.

Examples:
    python -m src.scripts.eval_inference run
    python -m src.scripts.eval_inference run --arrival-mode fixed-rate --request-rate 2
    python -m src.scripts.eval_inference compare path/to/baseline_summary.json path/to/candidate_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints" / "meta-llama" / "Meta-Llama-3.1-8B-Instruct"
DEFAULT_PROMPTS = [
    "Summarize what a KV cache does during autoregressive generation.",
    "Explain the difference between tensor parallelism and pipeline parallelism.",
    "Write a short Python function that returns the factorial of n.",
    "Give a concise explanation of why continuous batching improves throughput.",
    "List three tradeoffs between dense attention and paged attention.",
    "What is the purpose of a scheduler in an inference engine?",
    "Describe how prompt length can affect time to first token.",
    "Write two sentences about why benchmarking should separate TTFT and throughput.",
]
DEFAULT_PERCENTILES = [50.0, 90.0, 95.0, 99.0]
THROUGHPUT_METRIC_KEYS = {
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
}


@dataclass
class PromptSpec:
    prompt: str
    max_tokens: int | None = None
    ignore_eos: bool | None = None
    request_id: str | None = None
    source_index: int = 0
    source: str = "builtin"


@dataclass
class RequestTrace:
    request_id: str
    prompt: str
    prompt_tokens: int
    submit_ts: float
    source_index: int
    source: str
    request_ref: Any
    first_token_ts: float | None = None
    finish_ts: float | None = None
    last_token_ts: float | None = None
    itl_s_list: list[float] = field(default_factory=list)
    output_tokens_observed: int = 0
    status: str | None = None
    stop_reason: str | None = None

    def to_record(self) -> dict[str, Any]:
        output_tokens = getattr(self.request_ref, "num_completion_tokens", self.output_tokens_observed)
        total_tokens = getattr(self.request_ref, "num_tokens", self.prompt_tokens + output_tokens)
        ttl_s = None if self.finish_ts is None else self.finish_ts - self.submit_ts
        ttft_s = None if self.first_token_ts is None else self.first_token_ts - self.submit_ts
        tpot_s = None
        if ttl_s is not None and ttft_s is not None and output_tokens > 1:
            tpot_s = (ttl_s - ttft_s) / (output_tokens - 1)
        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "source": self.source,
            "source_index": self.source_index,
            "submit_ts": self.submit_ts,
            "first_token_ts": self.first_token_ts,
            "finish_ts": self.finish_ts,
            "ttft_s": ttft_s,
            "ttl_s": ttl_s,
            "e2el_s": ttl_s,
            "itl_s_list": list(self.itl_s_list),
            "tpot_s": tpot_s,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "status": self.status,
            "stop_reason": self.stop_reason,
        }


def _ensure_repo_imports() -> None:
    root_str = str(ROOT)
    litgpt_str = str(ROOT / "litgpt")
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if litgpt_str not in sys.path:
        sys.path.insert(0, litgpt_str)


def _parse_metric_percentiles(raw: str) -> list[float]:
    if not raw.strip():
        raise argparse.ArgumentTypeError("metric percentiles must not be empty")
    values: list[float] = []
    seen: set[float] = set()
    for part in raw.split(","):
        piece = part.strip()
        if not piece:
            continue
        try:
            percentile = float(piece)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid percentile {piece!r}") from exc
        if percentile < 0 or percentile > 100:
            raise argparse.ArgumentTypeError(
                f"percentile must be in [0, 100], got {percentile}"
            )
        if percentile in seen:
            continue
        seen.add(percentile)
        values.append(percentile)
    if not values:
        raise argparse.ArgumentTypeError("metric percentiles must not be empty")
    return sorted(values)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = percentile / 100.0 * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _build_metric_stats(values: list[float], percentiles: list[float]) -> dict[str, Any]:
    if values:
        mean_value = statistics.fmean(values)
        std_value = statistics.pstdev(values) if len(values) > 1 else 0.0
        median_value = statistics.median(values)
    else:
        mean_value = 0.0
        std_value = 0.0
        median_value = 0.0
    return {
        "count": len(values),
        "mean": mean_value,
        "std": std_value,
        "median": median_value,
        "percentiles": {f"p{_format_percentile_label(p)}": _percentile(values, p) for p in percentiles},
    }


def _format_percentile_label(percentile: float) -> str:
    return str(int(percentile)) if percentile.is_integer() else str(percentile)


def _load_prompt_specs(prompt_file: Path | None) -> list[PromptSpec]:
    if prompt_file is None:
        return [
            PromptSpec(prompt=prompt, source_index=index, source="builtin")
            for index, prompt in enumerate(DEFAULT_PROMPTS)
        ]

    specs: list[PromptSpec] = []
    with prompt_file.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {index + 1} of {prompt_file}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"line {index + 1} of {prompt_file} must be a JSON object")
            prompt = payload.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"line {index + 1} of {prompt_file} must contain a non-empty 'prompt'")

            max_tokens = payload.get("max_tokens")
            if max_tokens is not None:
                if not isinstance(max_tokens, int) or max_tokens < 1:
                    raise ValueError(
                        f"line {index + 1} of {prompt_file} has invalid 'max_tokens': {max_tokens!r}"
                    )

            ignore_eos = payload.get("ignore_eos")
            if ignore_eos is not None and not isinstance(ignore_eos, bool):
                raise ValueError(
                    f"line {index + 1} of {prompt_file} has invalid 'ignore_eos': {ignore_eos!r}"
                )

            request_id = payload.get("request_id")
            if request_id is not None and not isinstance(request_id, str):
                raise ValueError(
                    f"line {index + 1} of {prompt_file} has invalid 'request_id': {request_id!r}"
                )

            specs.append(
                PromptSpec(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    ignore_eos=ignore_eos,
                    request_id=request_id,
                    source_index=index,
                    source=str(prompt_file),
                )
            )

    if not specs:
        raise ValueError(f"no prompts found in {prompt_file}")
    return specs


def _select_prompt_specs(
    specs: list[PromptSpec],
    num_prompts: int | None,
    seed: int | None,
) -> list[PromptSpec]:
    if num_prompts is None or num_prompts >= len(specs):
        return list(specs)
    if num_prompts < 1:
        raise ValueError("num_prompts must be >= 1")
    if seed is None:
        return list(specs[:num_prompts])
    rng = random.Random(seed)
    indices = rng.sample(range(len(specs)), num_prompts)
    return [specs[index] for index in indices]


def _build_submission_offsets(
    count: int,
    arrival_mode: str,
    request_rate: float,
) -> list[float]:
    if arrival_mode == "burst":
        return [0.0] * count
    if arrival_mode == "fixed-rate":
        return [index / request_rate for index in range(count)]
    raise ValueError(f"unsupported arrival mode: {arrival_mode}")


def _observe_request(trace: RequestTrace, now_s: float) -> None:
    request = trace.request_ref
    output_tokens = getattr(request, "num_completion_tokens", trace.output_tokens_observed)
    new_tokens = output_tokens - trace.output_tokens_observed

    for _ in range(max(0, new_tokens)):
        if trace.first_token_ts is None:
            trace.first_token_ts = now_s
        elif trace.last_token_ts is not None:
            trace.itl_s_list.append(now_s - trace.last_token_ts)
        trace.last_token_ts = now_s

    trace.output_tokens_observed = output_tokens

    if trace.finish_ts is None and getattr(request, "is_finished")():
        trace.finish_ts = now_s
        trace.status = getattr(request, "status").name
        trace.stop_reason = getattr(request, "stop_reason", None)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _print_run_report(summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    print("\n============ Inference Benchmark Result ============")
    print("{:<40} {:<10}".format("Backend:", summary["config"]["kv_cache_manager"]))
    print("{:<40} {:<10}".format("Successful requests:", metrics["completed_requests"]))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", metrics["benchmark_duration_s"]))
    print("{:<40} {:<10}".format("Total input tokens:", metrics["total_input_tokens"]))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics["total_output_tokens"]))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics["request_throughput"]))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics["output_throughput"]))
    print("{:<40} {:<10.2f}".format("Total token throughput (tok/s):", metrics["total_token_throughput"]))
    _print_metric_section(metrics, "ttft", "TTFT", "Time to First Token")
    _print_metric_section(metrics, "tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    _print_metric_section(metrics, "itl", "ITL", "Inter-token Latency")
    _print_metric_section(metrics, "e2el", "E2EL", "End-to-end Latency")
    print("{:<40} {:<10}".format("Total engine steps:", metrics["total_engine_steps"]))
    print(
        "{:<40} {:<10.2f}".format(
            "Mean unfinished requests / step:",
            metrics["mean_unfinished_requests_per_step"],
        )
    )
    print(
        "{:<40} {:<10}".format(
            "Max unfinished requests / step:",
            metrics["max_unfinished_requests_per_step"],
        )
    )
    print("==================================================\n")


def _print_metric_section(
    metrics: dict[str, Any],
    name: str,
    label: str,
    header: str,
) -> None:
    print("{s:{c}^{n}}".format(s=header, n=50, c="-"))
    print("{:<40} {:<10.2f}".format(f"Mean {label} (ms):", metrics[f"mean_{name}_ms"]))
    print("{:<40} {:<10.2f}".format(f"Median {label} (ms):", metrics[f"median_{name}_ms"]))
    for percentile_label, value in metrics[f"{name}_percentiles_ms"].items():
        print("{:<40} {:<10.2f}".format(f"{percentile_label.upper()} {label} (ms):", value))


def _collect_metric_values(request_records: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in request_records:
        value = record.get(key)
        if value is None:
            continue
        values.append(float(value))
    return values


def _collect_itl_values(request_records: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for record in request_records:
        values.extend(float(item) for item in record.get("itl_s_list", []))
    return values


def _aggregate_metrics(
    request_records: list[dict[str, Any]],
    benchmark_duration_s: float,
    percentiles: list[float],
    total_engine_steps: int,
    unfinished_per_step: list[int],
) -> dict[str, Any]:
    completed_records = [record for record in request_records if record["finish_ts"] is not None]
    ttft_stats = _build_metric_stats(_collect_metric_values(completed_records, "ttft_s"), percentiles)
    ttl_stats = _build_metric_stats(_collect_metric_values(completed_records, "ttl_s"), percentiles)
    itl_stats = _build_metric_stats(_collect_itl_values(completed_records), percentiles)
    tpot_stats = _build_metric_stats(_collect_metric_values(completed_records, "tpot_s"), percentiles)

    total_input_tokens = sum(int(record["prompt_tokens"]) for record in completed_records)
    total_output_tokens = sum(int(record["output_tokens"]) for record in completed_records)
    duration = benchmark_duration_s if benchmark_duration_s > 0 else 1e-12
    mean_unfinished = statistics.fmean(unfinished_per_step) if unfinished_per_step else 0.0

    metrics = {
        "benchmark_duration_s": benchmark_duration_s,
        "completed_requests": len(completed_records),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "request_throughput": len(completed_records) / duration,
        "output_throughput": total_output_tokens / duration,
        "total_token_throughput": (total_input_tokens + total_output_tokens) / duration,
        "total_engine_steps": total_engine_steps,
        "mean_unfinished_requests_per_step": mean_unfinished,
        "max_unfinished_requests_per_step": max(unfinished_per_step) if unfinished_per_step else 0,
    }

    metrics.update(_flatten_metric_stats("ttft", ttft_stats))
    metrics.update(_flatten_metric_stats("ttl", ttl_stats))
    metrics.update(_flatten_metric_stats("e2el", ttl_stats))
    metrics.update(_flatten_metric_stats("itl", itl_stats))
    metrics.update(_flatten_metric_stats("tpot", tpot_stats))
    return metrics


def _flatten_metric_stats(name: str, stats: dict[str, Any]) -> dict[str, Any]:
    result = {
        f"{name}_count": stats["count"],
        f"mean_{name}_ms": stats["mean"] * 1000.0,
        f"std_{name}_ms": stats["std"] * 1000.0,
        f"median_{name}_ms": stats["median"] * 1000.0,
        f"{name}_percentiles_ms": {
            key: value * 1000.0 for key, value in stats["percentiles"].items()
        },
    }
    for percentile_label, value in stats["percentiles"].items():
        result[f"{percentile_label}_{name}_ms"] = value * 1000.0
    return result


def _make_output_paths(output_dir: Path, result_prefix: str) -> tuple[Path, Path]:
    prefix = result_prefix
    if prefix and not prefix.endswith(("-", "_")):
        prefix = f"{prefix}-"
    return output_dir / f"{prefix}summary.json", output_dir / f"{prefix}requests.jsonl"


def _write_requests_jsonl(path: Path, request_records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in request_records:
            handle.write(json.dumps(record, default=_json_default))
            handle.write("\n")


def _run_benchmark(args: argparse.Namespace) -> int:
    _ensure_repo_imports()
    from src.config.vllm import VllmConfig
    from src.engine.async_engine import AsyncEngine
    from src.sampling_params import SamplingParams

    prompt_specs = _select_prompt_specs(
        _load_prompt_specs(args.prompt_file),
        num_prompts=args.num_prompts,
        seed=args.seed,
    )
    submission_offsets = _build_submission_offsets(
        len(prompt_specs),
        arrival_mode=args.arrival_mode,
        request_rate=args.request_rate,
    )

    config = VllmConfig(
        checkpoint_dir=args.checkpoint_dir,
        kv_cache_manager=args.kv_cache_manager,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    init_start = time.perf_counter()
    engine = AsyncEngine(config)
    engine_init_s = time.perf_counter() - init_start

    traces: list[RequestTrace] = []
    next_submission_index = 0
    total_engine_steps = 0
    unfinished_per_step: list[int] = []

    benchmark_start = time.perf_counter()
    while next_submission_index < len(prompt_specs) or engine.has_unfinished_requests():
        elapsed_s = time.perf_counter() - benchmark_start
        while (
            next_submission_index < len(prompt_specs)
            and elapsed_s >= submission_offsets[next_submission_index]
        ):
            spec = prompt_specs[next_submission_index]
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=spec.max_tokens or args.max_tokens,
                ignore_eos=args.ignore_eos if spec.ignore_eos is None else spec.ignore_eos,
            )
            request = engine.add_request(
                prompt=spec.prompt,
                sampling_params=sampling_params,
                request_id=spec.request_id,
            )
            submit_ts = time.perf_counter() - benchmark_start
            traces.append(
                RequestTrace(
                    request_id=request.request_id,
                    prompt=spec.prompt,
                    prompt_tokens=request.num_prompt_tokens,
                    submit_ts=submit_ts,
                    source_index=spec.source_index,
                    source=spec.source,
                    request_ref=request,
                )
            )
            next_submission_index += 1
            elapsed_s = time.perf_counter() - benchmark_start

        if engine.has_unfinished_requests():
            unfinished_count = sum(1 for trace in traces if trace.finish_ts is None)
            unfinished_per_step.append(unfinished_count)
            engine.step()
            total_engine_steps += 1
            observe_ts = time.perf_counter() - benchmark_start
            for trace in traces:
                if trace.finish_ts is None:
                    _observe_request(trace, observe_ts)
            continue

        if next_submission_index < len(prompt_specs):
            wait_s = submission_offsets[next_submission_index] - elapsed_s
            if wait_s > 0:
                time.sleep(min(wait_s, 0.001))

    benchmark_duration_s = time.perf_counter() - benchmark_start
    request_records = [trace.to_record() for trace in traces]
    metrics = _aggregate_metrics(
        request_records=request_records,
        benchmark_duration_s=benchmark_duration_s,
        percentiles=args.metric_percentiles,
        total_engine_steps=total_engine_steps,
        unfinished_per_step=unfinished_per_step,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path, requests_path = _make_output_paths(output_dir, args.result_prefix)

    summary = {
        "created_at_utc": datetime.now(timezone.utc),
        "mode": "run",
        "config": {
            "checkpoint_dir": args.checkpoint_dir,
            "kv_cache_manager": args.kv_cache_manager,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_tokens": args.max_tokens,
            "ignore_eos": args.ignore_eos,
            "arrival_mode": args.arrival_mode,
            "request_rate": args.request_rate,
            "seed": args.seed,
            "metric_percentiles": args.metric_percentiles,
        },
        "workload": {
            "prompt_file": args.prompt_file,
            "num_requests": len(prompt_specs),
            "builtin_prompt_count": len(DEFAULT_PROMPTS) if args.prompt_file is None else 0,
        },
        "engine": {
            "engine_init_s": engine_init_s,
            "num_gpu_blocks": getattr(engine, "num_gpu_blocks", None),
            "max_seq_length": getattr(engine, "max_seq_length", None),
            "eos_token_id": getattr(engine, "eos_token_id", None),
        },
        "artifacts": {
            "summary_path": summary_path,
            "requests_path": requests_path,
        },
        "metrics": metrics,
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)
        handle.write("\n")
    _write_requests_jsonl(requests_path, request_records)

    _print_run_report(summary)
    print(f"summary.json:  {summary_path}")
    print(f"requests.jsonl:{requests_path}")
    return 0


def _load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path} is missing a 'metrics' object")
    return payload


def _metric_direction(metric_name: str) -> str:
    if metric_name in THROUGHPUT_METRIC_KEYS:
        return "higher"
    return "lower"


def _safe_delta_percent(baseline: float, candidate: float) -> float | None:
    if baseline == 0:
        return None
    return ((candidate - baseline) / baseline) * 100.0


def _format_delta(delta: float) -> str:
    return f"{delta:+.2f}"


def _format_delta_percent(delta_percent: float | None) -> str:
    if delta_percent is None:
        return "n/a"
    return f"{delta_percent:+.2f}%"


def _format_improvement(metric_name: str, baseline: float, candidate: float) -> str:
    direction = _metric_direction(metric_name)
    if baseline == candidate:
        return "flat"
    if direction == "higher":
        return "better" if candidate > baseline else "worse"
    return "better" if candidate < baseline else "worse"


def _iter_compare_metrics(baseline_metrics: dict[str, Any], candidate_metrics: dict[str, Any]) -> list[str]:
    preferred_order = [
        "completed_requests",
        "benchmark_duration_s",
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p90_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "mean_e2el_ms",
        "median_e2el_ms",
        "p90_e2el_ms",
        "p95_e2el_ms",
        "p99_e2el_ms",
        "total_engine_steps",
        "mean_unfinished_requests_per_step",
        "max_unfinished_requests_per_step",
    ]
    available = [key for key in preferred_order if key in baseline_metrics and key in candidate_metrics]
    return available


def _print_compare_report(
    baseline_path: Path,
    candidate_path: Path,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
) -> None:
    baseline_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]

    print("\n================ Benchmark Comparison ================")
    print(f"Baseline : {baseline_path}")
    print(f"Candidate: {candidate_path}")
    print("{:<34} {:>12} {:>12} {:>12} {:>12} {:>10}".format(
        "Metric",
        "Baseline",
        "Candidate",
        "Delta",
        "Delta %",
        "Verdict",
    ))
    for metric_name in _iter_compare_metrics(baseline_metrics, candidate_metrics):
        baseline_value = float(baseline_metrics[metric_name])
        candidate_value = float(candidate_metrics[metric_name])
        delta = candidate_value - baseline_value
        delta_percent = _safe_delta_percent(baseline_value, candidate_value)
        verdict = _format_improvement(metric_name, baseline_value, candidate_value)
        print(
            "{:<34} {:>12.2f} {:>12.2f} {:>12} {:>12} {:>10}".format(
                metric_name,
                baseline_value,
                candidate_value,
                _format_delta(delta),
                _format_delta_percent(delta_percent),
                verdict,
            )
        )
    print("======================================================\n")


def _compare_summaries(args: argparse.Namespace) -> int:
    baseline_summary = _load_summary(args.baseline_summary)
    candidate_summary = _load_summary(args.candidate_summary)
    _print_compare_report(
        baseline_path=args.baseline_summary,
        candidate_path=args.candidate_summary,
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference benchmark harness for mini-vLLM")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an inference benchmark")
    run_parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Path to the LitGPT checkpoint directory (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    run_parser.add_argument(
        "--kv-cache-manager",
        choices=("standard", "paged"),
        default="standard",
        help="KV cache backend to benchmark",
    )
    run_parser.add_argument("--max-num-seqs", type=_positive_int, default=2)
    run_parser.add_argument("--max-num-batched-tokens", type=_positive_int, default=16384)
    run_parser.add_argument("--max-model-len", type=_positive_int, default=8192)
    run_parser.add_argument("--gpu-memory-utilization", type=_positive_float, default=0.9)
    run_parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional JSONL file with a 'prompt' field and optional per-row overrides",
    )
    run_parser.add_argument(
        "--num-prompts",
        type=_positive_int,
        default=None,
        help="Limit the benchmark to this many prompts; with --seed, sample reproducibly",
    )
    run_parser.add_argument("--max-tokens", type=_positive_int, default=64)
    ignore_group = run_parser.add_mutually_exclusive_group()
    ignore_group.add_argument("--ignore-eos", dest="ignore_eos", action="store_true")
    ignore_group.add_argument("--respect-eos", dest="ignore_eos", action="store_false")
    run_parser.set_defaults(ignore_eos=True)
    run_parser.add_argument("--arrival-mode", choices=("burst", "fixed-rate"), default="burst")
    run_parser.add_argument(
        "--request-rate",
        type=_positive_float,
        default=1.0,
        help="Requests per second when --arrival-mode=fixed-rate",
    )
    run_parser.add_argument(
        "--seed",
        type=_non_negative_int,
        default=None,
        help="Seed used when sampling prompts with --num-prompts",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark_results",
        help="Directory for summary.json and requests.jsonl artifacts",
    )
    run_parser.add_argument(
        "--result-prefix",
        type=str,
        default=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Prefix prepended to artifact filenames",
    )
    run_parser.add_argument(
        "--metric-percentiles",
        type=_parse_metric_percentiles,
        default=list(DEFAULT_PERCENTILES),
        help="Comma-separated percentiles to report, e.g. 50,90,95,99",
    )
    run_parser.set_defaults(func=_run_benchmark)

    compare_parser = subparsers.add_parser("compare", help="Compare two benchmark summaries")
    compare_parser.add_argument("baseline_summary", type=Path)
    compare_parser.add_argument("candidate_summary", type=Path)
    compare_parser.set_defaults(func=_compare_summaries)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        if args.arrival_mode == "fixed-rate" and args.request_rate <= 0:
            parser.error("--request-rate must be > 0 when --arrival-mode=fixed-rate")
        if args.gpu_memory_utilization > 1.0:
            parser.error("--gpu-memory-utilization must be <= 1.0")
        if args.prompt_file is not None and not args.prompt_file.exists():
            parser.error(f"prompt file not found: {args.prompt_file}")

    if args.command == "compare":
        missing = [path for path in (args.baseline_summary, args.candidate_summary) if not path.exists()]
        if missing:
            parser.error("missing summary file(s): " + ", ".join(str(path) for path in missing))

    try:
        return args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
