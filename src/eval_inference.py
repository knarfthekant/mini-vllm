#!/usr/bin/env python3
"""
Inference benchmark harness for mini-vLLM.

This script measures request-level latency and throughput for the current dense
and paged backends. It can also sample a varied-length ShareGPT workload and
render a side-by-side TTFT/throughput comparison graph from two summaries.

Examples:
    python -m src.eval_inference run
    python -m src.eval_inference sharegpt-run --num-prompts 100
    python -m src.eval_inference compare path/to/baseline_summary.json path/to/candidate_summary.json
    python -m src.eval_inference graph path/to/baseline_summary.json path/to/candidate_summary.json
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

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
DEFAULT_SHAREGPT_DATASET = "Aeala/ShareGPT_Vicuna_unfiltered"
DEFAULT_SHAREGPT_SPLIT = "train"
DEFAULT_SHAREGPT_LENGTH_BUCKETS = 5
DEFAULT_SHAREGPT_OVERSAMPLE_FACTOR = 8
DEFAULT_SHAREGPT_MAX_ROWS = 5000
DEFAULT_SHAREGPT_MIN_PROMPT_TOKENS = 32
STEP_TIMING_KEYS = [
    "schedule_s",
    "prefinished_apply_s",
    "execute_model_s",
    "runner_update_states_s",
    "runner_prepare_inputs_s",
    "runner_forward_s",
    "runner_sample_s",
    "scheduler_postprocess_s",
    "finished_apply_s",
    "step_total_s",
]
THROUGHPUT_METRIC_KEYS = {
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
}
GRAPH_TTFT_METRICS = [
    ("mean_ttft_ms", "Mean", "ms"),
    ("median_ttft_ms", "Median", "ms"),
    ("p90_ttft_ms", "P90", "ms"),
    ("p95_ttft_ms", "P95", "ms"),
    ("p99_ttft_ms", "P99", "ms"),
]
GRAPH_THROUGHPUT_METRICS = [
    ("request_throughput", "Requests/s", "req/s"),
    ("output_throughput", "Output tok/s", "tok/s"),
    ("total_token_throughput", "Total tok/s", "tok/s"),
]


@dataclass
class PromptSpec:
    prompt: str
    max_tokens: int | None = None
    ignore_eos: bool | None = None
    request_id: str | None = None
    source_index: int = 0
    source: str = "builtin"
    prompt_tokens: int | None = None


@dataclass(frozen=True)
class ShareGPTPromptCandidate:
    prompt: str
    prompt_tokens: int
    source_index: int
    source_id: str
    conversation_turns: int


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


def _cap_prompt_token_ids(prompt_token_ids: list[int], prompt_token_cap: int | None) -> list[int]:
    if prompt_token_cap is None or len(prompt_token_ids) <= prompt_token_cap:
        return prompt_token_ids
    return prompt_token_ids[:prompt_token_cap]


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
        min_value = min(values)
        max_value = max(values)
    else:
        mean_value = 0.0
        std_value = 0.0
        median_value = 0.0
        min_value = 0.0
        max_value = 0.0
    return {
        "count": len(values),
        "mean": mean_value,
        "std": std_value,
        "median": median_value,
        "min": min_value,
        "max": max_value,
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

            prompt_tokens = payload.get("prompt_tokens")
            if prompt_tokens is not None:
                if not isinstance(prompt_tokens, int) or prompt_tokens < 1:
                    raise ValueError(
                        f"line {index + 1} of {prompt_file} has invalid 'prompt_tokens': {prompt_tokens!r}"
                    )

            specs.append(
                PromptSpec(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    ignore_eos=ignore_eos,
                    request_id=request_id,
                    source_index=index,
                    source=str(prompt_file),
                    prompt_tokens=prompt_tokens,
                )
            )

    if not specs:
        raise ValueError(f"no prompts found in {prompt_file}")
    return specs


def _write_prompt_specs_jsonl(path: Path, prompt_specs: Iterable[PromptSpec]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for spec in prompt_specs:
            payload: dict[str, Any] = {
                "prompt": spec.prompt,
                "source": spec.source,
                "source_index": spec.source_index,
            }
            if spec.max_tokens is not None:
                payload["max_tokens"] = spec.max_tokens
            if spec.ignore_eos is not None:
                payload["ignore_eos"] = spec.ignore_eos
            if spec.request_id is not None:
                payload["request_id"] = spec.request_id
            if spec.prompt_tokens is not None:
                payload["prompt_tokens"] = spec.prompt_tokens
            handle.write(json.dumps(payload, default=_json_default))
            handle.write("\n")


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
    print("{:<40} {:<10}".format("Mode:", summary["mode"]))
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
    _print_step_timing_section(summary)
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


def _print_step_timing_section(summary: dict[str, Any]) -> None:
    step_timing = summary.get("diagnostics", {}).get("step_timing")
    if not step_timing:
        return

    print("{s:{c}^{n}}".format(s="Step Timing", n=50, c="-"))
    print("{:<40} {:<10}".format("Timed engine steps:", step_timing["num_steps"]))
    print("{:<40} {:<10}".format("Timing mode:", step_timing["timing_mode"]))

    phase_specs = [
        ("Mean schedule (ms):", "schedule_s"),
        ("Mean input prep (ms):", "runner_prepare_inputs_s"),
        ("Mean forward launch (ms):", "runner_forward_s"),
        ("Mean sample sync (ms):", "runner_sample_s"),
        ("Mean postprocess (ms):", "scheduler_postprocess_s"),
        ("Mean full step (ms):", "step_total_s"),
    ]
    for label, key in phase_specs:
        stats = step_timing["phases"].get(key)
        if stats is None:
            continue
        print("{:<40} {:<10.2f}".format(label, stats["mean_ms"]))


def _print_sharegpt_sample_report(path: Path, workload: dict[str, Any]) -> None:
    stats = workload["sampled_prompt_token_stats"]
    print("\n================ ShareGPT Prompt Sample ================")
    print("{:<34} {}".format("Dataset:", workload["sharegpt_dataset"]))
    print("{:<34} {}".format("Split:", workload["sharegpt_split"]))
    print("{:<34} {}".format("Selected prompts:", workload["num_requests"]))
    print("{:<34} {}".format("Length buckets:", workload["sharegpt_length_buckets"]))
    print("{:<34} {}".format("Prompt token min:", int(stats["min"])))
    print("{:<34} {:.2f}".format("Prompt token mean:", stats["mean"]))
    print("{:<34} {:.2f}".format("Prompt token median:", stats["median"]))
    print("{:<34} {}".format("Prompt token max:", int(stats["max"])))
    print("{:<34} {}".format("Prompt file:", path))
    print("========================================================\n")


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


def _build_step_timing_stats(step_stats: list[dict[str, float]]) -> dict[str, Any] | None:
    if not step_stats:
        return None

    phases: dict[str, Any] = {}
    for key in STEP_TIMING_KEYS:
        values = [float(stats[key]) for stats in step_stats if key in stats]
        if not values:
            continue
        phases[key] = {
            "mean_ms": statistics.fmean(values) * 1000.0,
            "median_ms": statistics.median(values) * 1000.0,
            "max_ms": max(values) * 1000.0,
            "total_s": sum(values),
        }

    return {
        "num_steps": len(step_stats),
        "timing_mode": "host_wall_time",
        "phases": phases,
    }


def _make_output_paths(output_dir: Path, result_prefix: str) -> tuple[Path, Path]:
    prefix = result_prefix
    if prefix and not prefix.endswith(("-", "_")):
        prefix = f"{prefix}-"
    return output_dir / f"{prefix}summary.json", output_dir / f"{prefix}requests.jsonl"


def _make_artifact_path(output_dir: Path, result_prefix: str, suffix: str) -> Path:
    prefix = result_prefix
    if prefix and not prefix.endswith(("-", "_")):
        prefix = f"{prefix}-"
    return output_dir / f"{prefix}{suffix}"


def _write_requests_jsonl(path: Path, request_records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in request_records:
            handle.write(json.dumps(record, default=_json_default))
            handle.write("\n")


def _first_non_empty_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def _normalize_prompt_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").splitlines()]
    return "\n".join(lines).strip()


def _build_sharegpt_prompt(conversations: Any) -> tuple[str, int] | None:
    if not isinstance(conversations, list):
        return None

    cleaned_turns: list[tuple[str, str]] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        speaker = turn.get("from")
        if speaker not in {"human", "gpt"}:
            continue
        content = _first_non_empty_text(turn.get("value"), turn.get("text"), turn.get("markdown"))
        if content is None:
            continue
        content = _normalize_prompt_text(content)
        if not content:
            continue
        if cleaned_turns and cleaned_turns[-1][0] == speaker:
            merged = f"{cleaned_turns[-1][1]}\n\n{content}"
            cleaned_turns[-1] = (speaker, merged)
            continue
        cleaned_turns.append((speaker, content))

    while cleaned_turns and cleaned_turns[0][0] != "human":
        cleaned_turns.pop(0)
    while cleaned_turns and cleaned_turns[-1][0] != "human":
        cleaned_turns.pop()

    if not cleaned_turns:
        return None

    lines = []
    for speaker, content in cleaned_turns:
        role = "User" if speaker == "human" else "Assistant"
        lines.append(f"{role}: {content}")
    lines.append("Assistant:")
    return "\n\n".join(lines), len(cleaned_turns)


def _build_prompt_length_counter(checkpoint_dir: Path) -> Any:
    checkpoint_dir = Path(checkpoint_dir)
    tokenizer_json = checkpoint_dir / "tokenizer.json"
    tokenizer_model = checkpoint_dir / "tokenizer.model"

    if tokenizer_json.is_file():
        try:
            from tokenizers import Tokenizer as HFTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "prompt sampling requires the 'tokenizers' package when tokenizer.json is present"
            ) from exc

        processor = HFTokenizer.from_file(str(tokenizer_json))
        return lambda text: len(processor.encode(text).ids)

    if tokenizer_model.is_file():
        try:
            from sentencepiece import SentencePieceProcessor
        except ImportError as exc:
            raise RuntimeError(
                "prompt sampling requires the 'sentencepiece' package when tokenizer.model is present"
            ) from exc

        processor = SentencePieceProcessor(model_file=str(tokenizer_model))
        return lambda text: len(processor.encode(text))

    raise RuntimeError(f"no tokenizer.json or tokenizer.model found in {checkpoint_dir}")


def _encode_token_count(tokenizer: Any, text: str) -> int:
    if callable(tokenizer) and not hasattr(tokenizer, "encode"):
        return int(tokenizer(text))
    encoded = tokenizer.encode(text)
    if hasattr(encoded, "numel"):
        return int(encoded.numel())
    return len(encoded)


def _resolve_sharegpt_max_prompt_tokens(args: argparse.Namespace) -> int:
    max_prompt_tokens = args.sharegpt_max_prompt_tokens
    budget_from_model_len = max(1, args.max_model_len - args.max_tokens)
    if max_prompt_tokens is None:
        max_prompt_tokens = budget_from_model_len
    else:
        max_prompt_tokens = min(max_prompt_tokens, budget_from_model_len)
    if args.prompt_token_cap is not None:
        max_prompt_tokens = min(max_prompt_tokens, args.prompt_token_cap)
    if max_prompt_tokens < args.sharegpt_min_prompt_tokens:
        raise ValueError(
            "sharegpt prompt-token range is empty; "
            f"min={args.sharegpt_min_prompt_tokens}, max={max_prompt_tokens}. "
            "Increase --max-model-len, decrease --max-tokens, or lower --sharegpt-min-prompt-tokens."
        )
    return max_prompt_tokens


def _collect_sharegpt_candidates(
    args: argparse.Namespace,
    tokenizer: Any,
) -> tuple[list[ShareGPTPromptCandidate], int]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "sharegpt sampling requires the 'datasets' package. "
            "Install it with: python -m pip install datasets==3.6.0"
        ) from exc

    max_prompt_tokens = _resolve_sharegpt_max_prompt_tokens(args)
    dataset_stream = load_dataset(
        args.sharegpt_dataset,
        name=args.sharegpt_config,
        split=args.sharegpt_split,
        streaming=True,
    )

    target_candidates = max(
        args.num_prompts,
        args.num_prompts * args.sharegpt_oversample_factor,
    )
    candidates: list[ShareGPTPromptCandidate] = []
    seen_prompts: set[str] = set()

    scanned_rows = 0
    for row_index, row in enumerate(dataset_stream):
        scanned_rows = row_index + 1
        if row_index >= args.sharegpt_max_rows:
            break

        prompt_data = _build_sharegpt_prompt(row.get("conversations"))
        if prompt_data is None:
            continue
        prompt, conversation_turns = prompt_data
        if prompt in seen_prompts:
            continue

        prompt_tokens = _encode_token_count(tokenizer, prompt)
        if prompt_tokens < args.sharegpt_min_prompt_tokens or prompt_tokens > max_prompt_tokens:
            continue

        seen_prompts.add(prompt)
        candidates.append(
            ShareGPTPromptCandidate(
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                source_index=row_index,
                source_id=str(row.get("id", row_index)),
                conversation_turns=conversation_turns,
            )
        )
        if len(candidates) >= target_candidates:
            break

    if len(candidates) < args.num_prompts:
        raise ValueError(
            "unable to collect enough ShareGPT prompts "
            f"(needed {args.num_prompts}, found {len(candidates)} after scanning {min(args.sharegpt_max_rows, scanned_rows)} rows). "
            "Increase --sharegpt-max-rows, widen the prompt-token range, or lower --num-prompts."
        )

    return candidates, max_prompt_tokens


def _bucket_sharegpt_candidates(
    candidates: list[ShareGPTPromptCandidate],
    bucket_count: int,
) -> list[list[ShareGPTPromptCandidate]]:
    sorted_candidates = sorted(candidates, key=lambda item: item.prompt_tokens)
    buckets: list[list[ShareGPTPromptCandidate]] = [[] for _ in range(bucket_count)]
    for index, candidate in enumerate(sorted_candidates):
        bucket_index = min(bucket_count - 1, index * bucket_count // len(sorted_candidates))
        buckets[bucket_index].append(candidate)
    return [bucket for bucket in buckets if bucket]


def _select_sharegpt_candidates(
    candidates: list[ShareGPTPromptCandidate],
    target_count: int,
    bucket_count: int,
    seed: int | None,
) -> list[ShareGPTPromptCandidate]:
    rng = random.Random(seed)
    buckets = _bucket_sharegpt_candidates(candidates, bucket_count)
    if not buckets:
        raise ValueError("no ShareGPT candidates available after filtering")

    allocation = [0] * len(buckets)
    remaining = target_count

    while remaining > 0:
        progressed = False
        for bucket_index, bucket in enumerate(buckets):
            if allocation[bucket_index] >= len(bucket):
                continue
            allocation[bucket_index] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break

    selected: list[ShareGPTPromptCandidate] = []
    for bucket, count in zip(buckets, allocation):
        if count <= 0:
            continue
        selected.extend(rng.sample(bucket, count))

    if len(selected) != target_count:
        raise ValueError(
            f"failed to allocate {target_count} ShareGPT prompts across {len(buckets)} buckets"
        )

    rng.shuffle(selected)
    return selected


def _sample_sharegpt_prompt_specs(
    args: argparse.Namespace,
    tokenizer: Any,
) -> tuple[list[PromptSpec], dict[str, Any]]:
    candidates, max_prompt_tokens = _collect_sharegpt_candidates(args, tokenizer)
    selected = _select_sharegpt_candidates(
        candidates=candidates,
        target_count=args.num_prompts,
        bucket_count=args.sharegpt_length_buckets,
        seed=args.seed,
    )

    source_name = f"{args.sharegpt_dataset}:{args.sharegpt_split}"
    prompt_specs = [
        PromptSpec(
            prompt=item.prompt,
            source_index=item.source_index,
            source=source_name,
            prompt_tokens=item.prompt_tokens,
        )
        for item in selected
    ]
    prompt_lengths = [float(item.prompt_tokens) for item in selected]
    workload = {
        "prompt_source": "sharegpt",
        "builtin_prompt_count": 0,
        "sharegpt_dataset": args.sharegpt_dataset,
        "sharegpt_config": args.sharegpt_config,
        "sharegpt_split": args.sharegpt_split,
        "sharegpt_min_prompt_tokens": args.sharegpt_min_prompt_tokens,
        "sharegpt_max_prompt_tokens": max_prompt_tokens,
        "sharegpt_length_buckets": args.sharegpt_length_buckets,
        "sharegpt_oversample_factor": args.sharegpt_oversample_factor,
        "sharegpt_max_rows": args.sharegpt_max_rows,
        "sharegpt_candidates_considered": len(candidates),
        "sampled_prompt_token_stats": _build_metric_stats(prompt_lengths, args.metric_percentiles),
    }
    return prompt_specs, workload


def _run_benchmark_with_prompt_specs(
    args: argparse.Namespace,
    prompt_specs: list[PromptSpec],
    *,
    mode: str,
    prompt_file_path: Path | None,
    workload_overrides: dict[str, Any] | None = None,
) -> int:
    _ensure_repo_imports()
    from src.config.vllm import VllmConfig
    from src.engine.async_engine import AsyncEngine
    from src.sampling_params import SamplingParams

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
    step_stats: list[dict[str, float]] = []

    benchmark_start = time.perf_counter()
    while next_submission_index < len(prompt_specs) or engine.has_unfinished_requests():
        elapsed_s = time.perf_counter() - benchmark_start
        while (
            next_submission_index < len(prompt_specs)
            and elapsed_s >= submission_offsets[next_submission_index]
        ):
            spec = prompt_specs[next_submission_index]
            prompt_input: str | list[int] = spec.prompt
            if args.prompt_token_cap is not None:
                prompt_input = _cap_prompt_token_ids(
                    engine.tokenizer.encode(spec.prompt).tolist(),
                    args.prompt_token_cap,
                )
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=spec.max_tokens or args.max_tokens,
                ignore_eos=args.ignore_eos if spec.ignore_eos is None else spec.ignore_eos,
            )
            request = engine.add_request(
                prompt=prompt_input,
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
            if args.profile_engine_step and engine.last_step_stats is not None:
                step_stats.append(dict(engine.last_step_stats))

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

    workload = {
        "prompt_file": prompt_file_path,
        "num_requests": len(prompt_specs),
        "builtin_prompt_count": len(DEFAULT_PROMPTS) if prompt_file_path is None else 0,
        "prompt_token_cap": args.prompt_token_cap,
    }
    if workload_overrides:
        workload.update(workload_overrides)

    diagnostics: dict[str, Any] = {}
    step_timing = _build_step_timing_stats(step_stats)
    if step_timing is not None:
        diagnostics["step_timing"] = step_timing

    summary = {
        "created_at_utc": datetime.now(timezone.utc),
        "mode": mode,
        "config": {
            "checkpoint_dir": args.checkpoint_dir,
            "kv_cache_manager": args.kv_cache_manager,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_model_len": args.max_model_len,
            "prompt_token_cap": args.prompt_token_cap,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_tokens": args.max_tokens,
            "ignore_eos": args.ignore_eos,
            "arrival_mode": args.arrival_mode,
            "request_rate": args.request_rate,
            "seed": args.seed,
            "metric_percentiles": args.metric_percentiles,
        },
        "workload": workload,
        "engine": {
            "engine_init_s": engine_init_s,
            "num_gpu_blocks": getattr(engine, "num_gpu_blocks", None),
            "max_seq_length": getattr(engine, "max_seq_length", None),
            "eos_token_id": getattr(engine, "eos_token_id", None),
        },
        "artifacts": {
            "summary_path": summary_path,
            "requests_path": requests_path,
            "prompt_file": prompt_file_path,
        },
        "metrics": metrics,
        "diagnostics": diagnostics,
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)
        handle.write("\n")
    _write_requests_jsonl(requests_path, request_records)

    _print_run_report(summary)
    print(f"summary.json:  {summary_path}")
    print(f"requests.jsonl:{requests_path}")
    if prompt_file_path is not None:
        print(f"prompt_file:   {prompt_file_path}")
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    prompt_specs = _select_prompt_specs(
        _load_prompt_specs(args.prompt_file),
        num_prompts=args.num_prompts,
        seed=args.seed,
    )
    return _run_benchmark_with_prompt_specs(
        args,
        prompt_specs,
        mode="run",
        prompt_file_path=args.prompt_file,
    )


def _prepare_sharegpt_prompts(args: argparse.Namespace) -> int:
    token_counter = _build_prompt_length_counter(args.checkpoint_dir)
    prompt_specs, workload = _sample_sharegpt_prompt_specs(args, token_counter)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = _make_artifact_path(args.output_dir, args.result_prefix, "prompts.jsonl")
    _write_prompt_specs_jsonl(prompt_path, prompt_specs)
    workload["num_requests"] = len(prompt_specs)
    _print_sharegpt_sample_report(prompt_path, workload)
    return 0


def _run_sharegpt_benchmark(args: argparse.Namespace) -> int:
    token_counter = _build_prompt_length_counter(args.checkpoint_dir)
    prompt_specs, workload = _sample_sharegpt_prompt_specs(args, token_counter)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = _make_artifact_path(args.output_dir, args.result_prefix, "prompts.jsonl")
    _write_prompt_specs_jsonl(prompt_path, prompt_specs)
    workload["num_requests"] = len(prompt_specs)
    _print_sharegpt_sample_report(prompt_path, workload)

    return _run_benchmark_with_prompt_specs(
        args,
        prompt_specs,
        mode="sharegpt-run",
        prompt_file_path=prompt_path,
        workload_overrides=workload,
    )


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
    return [key for key in preferred_order if key in baseline_metrics and key in candidate_metrics]


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


def _format_graph_value(value: float, unit: str) -> str:
    return f"{value:.2f} {unit}".rstrip()


def _svg_rect(x: float, y: float, width: float, height: float, fill: str, stroke: str | None = None) -> str:
    attrs = [f'x="{x:.2f}"', f'y="{y:.2f}"', f'width="{width:.2f}"', f'height="{height:.2f}"', f'fill="{fill}"']
    if stroke is not None:
        attrs.append(f'stroke="{stroke}"')
    return f"<rect {' '.join(attrs)} />"


def _svg_text(
    x: float,
    y: float,
    text: str,
    *,
    font_size: int = 14,
    fill: str = "#111827",
    anchor: str = "start",
    weight: str = "400",
) -> str:
    escaped = html.escape(text, quote=True)
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{font_size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escaped}</text>'
    )


def _render_metric_panel(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    metric_specs: list[tuple[str, str, str]],
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    baseline_label: str,
    candidate_label: str,
) -> list[str]:
    elements = [
        _svg_rect(x, y, width, height, fill="#ffffff", stroke="#d1d5db"),
        _svg_text(x + 24, y + 34, title, font_size=22, weight="700"),
        _svg_text(x + 24, y + 58, "Each row is scaled to its own local max.", font_size=12, fill="#6b7280"),
    ]

    label_x = x + 24
    series_label_x = x + 164
    bar_x = x + 180
    bar_width = width - 310
    value_x = x + width - 20
    row_top = y + 84
    row_height = (height - 110) / max(1, len(metric_specs))
    rail_color = "#e5e7eb"
    baseline_color = "#2563eb"
    candidate_color = "#ef4444"

    for row_index, (metric_key, metric_label, row_unit) in enumerate(metric_specs):
        base_value = float(baseline_metrics.get(metric_key, 0.0))
        cand_value = float(candidate_metrics.get(metric_key, 0.0))
        local_max = max(base_value, cand_value, 1e-9)
        section_y = row_top + row_index * row_height

        elements.append(_svg_text(label_x, section_y + 16, metric_label, font_size=15, weight="700"))

        series = [
            (baseline_label, base_value, baseline_color, section_y + 34),
            (candidate_label, cand_value, candidate_color, section_y + 60),
        ]
        for series_label, value, color, bar_y in series:
            elements.append(_svg_text(series_label_x, bar_y + 12, series_label, font_size=12, fill="#374151", anchor="end"))
            elements.append(_svg_rect(bar_x, bar_y, bar_width, 14, fill=rail_color))
            fill_width = 0.0 if value <= 0 else max(2.0, bar_width * (value / local_max))
            elements.append(_svg_rect(bar_x, bar_y, fill_width, 14, fill=color))
            elements.append(
                _svg_text(
                    value_x,
                    bar_y + 12,
                    _format_graph_value(value, row_unit),
                    font_size=12,
                    fill="#111827",
                    anchor="end",
                )
            )

    return elements


def _render_comparison_graph_svg(
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    *,
    baseline_label: str,
    candidate_label: str,
    baseline_path: Path,
    candidate_path: Path,
) -> str:
    width = 1420
    height = 780
    margin = 36
    panel_gap = 24
    panel_width = (width - margin * 2 - panel_gap) / 2
    panel_height = height - 180
    panel_y = 136

    baseline_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]

    elements = [
        _svg_rect(0, 0, width, height, fill="#f8fafc"),
        _svg_text(margin, 42, "mini-vLLM Benchmark Comparison", font_size=30, weight="700"),
        _svg_text(
            margin,
            72,
            f"Baseline: {baseline_path.name}    Candidate: {candidate_path.name}",
            font_size=14,
            fill="#475569",
        ),
        _svg_rect(margin, 92, 14, 14, fill="#2563eb"),
        _svg_text(margin + 22, 104, baseline_label, font_size=13, fill="#334155"),
        _svg_rect(margin + 130, 92, 14, 14, fill="#ef4444"),
        _svg_text(margin + 152, 104, candidate_label, font_size=13, fill="#334155"),
    ]

    elements.extend(
        _render_metric_panel(
            margin,
            panel_y,
            panel_width,
            panel_height,
            title="TTFT Comparison",
            metric_specs=GRAPH_TTFT_METRICS,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            baseline_label=baseline_label,
            candidate_label=candidate_label,
        )
    )
    elements.extend(
        _render_metric_panel(
            margin + panel_width + panel_gap,
            panel_y,
            panel_width,
            panel_height,
            title="Throughput Comparison",
            metric_specs=GRAPH_THROUGHPUT_METRICS,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            baseline_label=baseline_label,
            candidate_label=candidate_label,
        )
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        + "".join(elements)
        + "</svg>\n"
    )


def _default_graph_output_path(baseline_summary: Path, candidate_summary: Path) -> Path:
    baseline_name = baseline_summary.stem.replace("-summary", "")
    candidate_name = candidate_summary.stem.replace("-summary", "")
    return baseline_summary.parent / f"{baseline_name}-vs-{candidate_name}-comparison.svg"


def _graph_summaries(args: argparse.Namespace) -> int:
    baseline_summary = _load_summary(args.baseline_summary)
    candidate_summary = _load_summary(args.candidate_summary)

    output_path = args.output or _default_graph_output_path(args.baseline_summary, args.candidate_summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    svg = _render_comparison_graph_svg(
        baseline_summary,
        candidate_summary,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        baseline_path=args.baseline_summary,
        candidate_path=args.candidate_summary,
    )
    output_path.write_text(svg, encoding="utf-8")
    print(f"graph.svg:     {output_path}")
    return 0


def _add_run_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_num_prompts: int | None,
    default_result_prefix: str,
) -> None:
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Path to the LitGPT checkpoint directory (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--kv-cache-manager",
        choices=("standard", "paged"),
        default="standard",
        help="KV cache backend to benchmark",
    )
    parser.add_argument("--max-num-seqs", type=_positive_int, default=5)
    parser.add_argument("--max-num-batched-tokens", type=_positive_int, default=16384)
    parser.add_argument("--max-model-len", type=_positive_int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=_positive_float, default=0.9)
    parser.add_argument(
        "--num-prompts",
        type=_positive_int,
        default=default_num_prompts,
        help="Prompt count to benchmark; with --seed, sample reproducibly",
    )
    parser.add_argument("--max-tokens", type=_positive_int, default=64)
    ignore_group = parser.add_mutually_exclusive_group()
    ignore_group.add_argument("--ignore-eos", dest="ignore_eos", action="store_true")
    ignore_group.add_argument("--respect-eos", dest="ignore_eos", action="store_false")
    parser.set_defaults(ignore_eos=True)
    parser.add_argument("--arrival-mode", choices=("burst", "fixed-rate"), default="burst")
    parser.add_argument(
        "--request-rate",
        type=_positive_float,
        default=1.0,
        help="Requests per second when --arrival-mode=fixed-rate",
    )
    parser.add_argument(
        "--seed",
        type=_non_negative_int,
        default=None,
        help="Seed used when sampling prompts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark_results",
        help="Directory for output artifacts",
    )
    parser.add_argument(
        "--result-prefix",
        type=str,
        default=default_result_prefix,
        help="Prefix prepended to artifact filenames",
    )
    parser.add_argument(
        "--metric-percentiles",
        type=_parse_metric_percentiles,
        default=list(DEFAULT_PERCENTILES),
        help="Comma-separated percentiles to report, e.g. 50,90,95,99",
    )
    parser.add_argument(
        "--prompt-token-cap",
        type=_positive_int,
        default=None,
        help="Optional cap on prompt tokens applied at submission time, e.g. 512",
    )
    parser.add_argument(
        "--profile-engine-step",
        action="store_true",
        help="Collect per-step host timing for scheduler, input prep, forward launch, and postprocess",
    )


def _add_sharegpt_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sharegpt-dataset",
        type=str,
        default=DEFAULT_SHAREGPT_DATASET,
        help="Hugging Face dataset path used for ShareGPT sampling",
    )
    parser.add_argument(
        "--sharegpt-config",
        type=str,
        default=None,
        help="Optional dataset config name",
    )
    parser.add_argument(
        "--sharegpt-split",
        type=str,
        default=DEFAULT_SHAREGPT_SPLIT,
        help="Dataset split used for sampling",
    )
    parser.add_argument(
        "--sharegpt-min-prompt-tokens",
        type=_positive_int,
        default=DEFAULT_SHAREGPT_MIN_PROMPT_TOKENS,
        help="Minimum prompt-token length after formatting the conversation",
    )
    parser.add_argument(
        "--sharegpt-max-prompt-tokens",
        type=_positive_int,
        default=None,
        help="Optional upper bound on prompt tokens; capped by max_model_len - max_tokens",
    )
    parser.add_argument(
        "--sharegpt-length-buckets",
        type=_positive_int,
        default=DEFAULT_SHAREGPT_LENGTH_BUCKETS,
        help="Number of prompt-length buckets used for stratified sampling",
    )
    parser.add_argument(
        "--sharegpt-oversample-factor",
        type=_positive_int,
        default=DEFAULT_SHAREGPT_OVERSAMPLE_FACTOR,
        help="How many valid candidates to collect relative to the target sample count",
    )
    parser.add_argument(
        "--sharegpt-max-rows",
        type=_positive_int,
        default=DEFAULT_SHAREGPT_MAX_ROWS,
        help="Maximum streamed dataset rows to scan before giving up",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference benchmark harness for mini-vLLM")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an inference benchmark")
    _add_run_arguments(
        run_parser,
        default_num_prompts=None,
        default_result_prefix=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    run_parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional JSONL file with a 'prompt' field and optional per-row overrides",
    )
    run_parser.set_defaults(func=_run_benchmark)

    sharegpt_prepare_parser = subparsers.add_parser(
        "sharegpt-prepare",
        help="Sample a varied-length ShareGPT prompt file",
    )
    _add_run_arguments(
        sharegpt_prepare_parser,
        default_num_prompts=100,
        default_result_prefix=f"sharegpt-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    _add_sharegpt_arguments(sharegpt_prepare_parser)
    sharegpt_prepare_parser.set_defaults(func=_prepare_sharegpt_prompts)

    sharegpt_run_parser = subparsers.add_parser(
        "sharegpt-run",
        help="Sample ShareGPT prompts and run the benchmark end-to-end",
    )
    _add_run_arguments(
        sharegpt_run_parser,
        default_num_prompts=100,
        default_result_prefix=f"sharegpt-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    _add_sharegpt_arguments(sharegpt_run_parser)
    sharegpt_run_parser.set_defaults(func=_run_sharegpt_benchmark)

    compare_parser = subparsers.add_parser("compare", help="Compare two benchmark summaries")
    compare_parser.add_argument("baseline_summary", type=Path)
    compare_parser.add_argument("candidate_summary", type=Path)
    compare_parser.set_defaults(func=_compare_summaries)

    graph_parser = subparsers.add_parser(
        "graph",
        help="Render an SVG with side-by-side TTFT and throughput comparison panels",
    )
    graph_parser.add_argument("baseline_summary", type=Path)
    graph_parser.add_argument("candidate_summary", type=Path)
    graph_parser.add_argument("--output", type=Path, default=None)
    graph_parser.add_argument("--baseline-label", type=str, default="Baseline")
    graph_parser.add_argument("--candidate-label", type=str, default="Candidate")
    graph_parser.set_defaults(func=_graph_summaries)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {"run", "sharegpt-run", "sharegpt-prepare"}:
        if args.arrival_mode == "fixed-rate" and args.request_rate <= 0:
            parser.error("--request-rate must be > 0 when --arrival-mode=fixed-rate")
        if args.gpu_memory_utilization > 1.0:
            parser.error("--gpu-memory-utilization must be <= 1.0")

    if args.command == "run":
        if args.prompt_file is not None and not args.prompt_file.exists():
            parser.error(f"prompt file not found: {args.prompt_file}")

    if args.command in {"sharegpt-run", "sharegpt-prepare"}:
        if args.num_prompts is None:
            parser.error("--num-prompts is required for ShareGPT commands")
        if args.sharegpt_length_buckets > args.num_prompts:
            parser.error("--sharegpt-length-buckets must be <= --num-prompts")

    if args.command in {"compare", "graph"}:
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
