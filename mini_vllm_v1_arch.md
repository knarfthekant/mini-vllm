# vLLM v1 Architecture Report: A Guide for Minimal Implementation

This report details the architecture of vLLM v1 based on source code analysis of the `vllm/v1` directory. It focuses on the components necessary to understand the execution flow and implement a simplified inference engine.

## 1. High-Level Architecture Overview

The vLLM v1 architecture is designed for high-performance, asynchronous LLM inference. It follows a layered approach, separating request management, scheduling, hardware orchestration, and model execution.

### Core Components and Responsibilities

| Component | Responsibility | Relevant Files |
| :--- | :--- | :--- |
| **AsyncLLM / EngineCore** | The entry point and main event loop. Orchestrates the flow between the Scheduler and Executor. | [engine/async_llm.py](file:///home/frank/dev/mini-vllm/src/engine/async_llm.py), [engine/core.py](file:///home/frank/dev/mini-vllm/vllm/vllm/v1/engine/core.py) |
| **Scheduler** | Manages request queues (Waiting, Running, Swapped-out). Implements continuous batching and preemptive scheduling logic. | `scheduler/scheduler.py` |
| **Executor** | Acts as the "Hardware Topology Orchestrator." Abstractions for single-process, multi-process, or distributed execution. | [executor/abstract.py](file:///home/frank/dev/mini-vllm/vllm/vllm/v1/executor/abstract.py), [executor/uniproc_executor.py](file:///home/frank/dev/mini-vllm/vllm/vllm/v1/executor/uniproc_executor.py) |
| **ModelRunner** | Handles GPU-specific preparations, input tensor construction, and the actual model forward pass. | [worker/gpu_model_runner.py](file:///home/frank/dev/mini-vllm/vllm/vllm/v1/worker/gpu_model_runner.py) |

---

## 2. Component Deep Dive

### A. EngineCore: The Orchestrator
The `EngineCore.step()` method is the heart of the engine. It runs in a loop and performs the following sequence:

1.  **Schedule**: Calls `Scheduler.schedule()` to determine which requests (and tokens) to run in this step.
2.  **Execute**: Passes the `SchedulerOutput` to the `Executor.execute_model()`.
3.  **Process Output**: Receives [ModelRunnerOutput](file:///home/frank/dev/mini-vllm/vllm/vllm/v1/worker/gpu_model_runner.py#219-285), updates request states (decoded tokens), and notifies the Scheduler of changes.

### B. Scheduler: Continuous Batching Logic
The Scheduler in v1 is more granular than in v0. It doesn't just manage requests; it manages "scheduled tokens."

-   **Continuous Batching**: It promotes requests from the [waiting](file:///home/frank/dev/mini-vllm/vllm/vllm/v1/core/sched/scheduler.py#1538-1543) queue to the [running](file:///home/frank/dev/mini-vllm/vllm/vllm/v1/engine/async_llm.py#1017-1021) queue as soon as KV cache slots are available.
-   **Preemption**: If the KV cache is full, it can "preempt" lower-priority requests, swapping their KV cache out to CPU memory.
-   **Output**: Produces a `SchedulerOutput` object containing:
    -   `scheduled_req_ids`: Which requests are active.
    -   `num_scheduled_tokens`: How many tokens to process for each request (supporting chunked prefill).

### C. Executor: Hardware Abstraction
The Executor separates the engine logic from the physical hardware layout.

-   **Role**: It is *not* a model wrapper. It manages Worker processes, handles collective RPCs (in TP/PP setups), and synchronizes data across GPUs.
-   **[execute_model](file:///home/frank/dev/mini-vllm/vllm/vllm/v1/executor/multiproc_executor.py#296-307)**: This is the primary method called by the Engine. It broadcasts the scheduling decision to all Workers.

### D. GPUModelRunner: The Execution Engine
This is where the most complex GPU-side logic resides.

#### 1. State Management ([_update_states](file:///home/frank/dev/mini-vllm/vllm/vllm/v1/worker/gpu_model_runner.py#1019-1323))
Before every forward pass, the [ModelRunner](file:///home/frank/dev/mini-vllm/vllm/vllm/v1/worker/gpu_model_runner.py#386-6722) updates its internal mapping of requests to KV cache slots based on the `SchedulerOutput`.

#### 2. Input Preparation ([_prepare_inputs](file:///home/frank/dev/mini-vllm/vllm/vllm/v1/worker/gpu_model_runner.py#1667-1885))
This is the most critical part for a minimal implementation. It constructs the input tensors for the model:
-   **[input_ids](file:///home/frank/dev/mini-vllm/vllm/vllm/v1/worker/gpu_model_runner.py#4901-4908)**: The token IDs to process.
-   **[positions](file:///home/frank/dev/mini-vllm/vllm/vllm/v1/worker/gpu_model_runner.py#895-908)**: The cumulative positional indices for each token.
-   **`attn_metadata`**: Metadata describing the KV cache layout (block tables, slot mappings) for the attention kernels (e.g., PagedAttention).

#### 3. Model Forward Pass
The actual call to the model happens in [_model_forward](file:///home/frank/dev/mini-vllm/vllm/vllm/v1/worker/gpu_model_runner.py#3258-3289) (triggered within [execute_model](file:///home/frank/dev/mini-vllm/vllm/vllm/v1/executor/multiproc_executor.py#296-307)).
```python
# Simplified flow in gpu_model_runner.py
def execute_model(self, scheduler_output):
    # 1. Update KV cache mappings and request states
    self._update_states(scheduler_output)
    
    # 2. Construct idx, positions, and attention metadata
    input_ids, positions, attn_metadata = self._prepare_inputs(scheduler_output)
    
    # 3. Call the model
    hidden_states = self.model(input_ids, positions, attn_metadata)
    
    # 4. Compute logits and sample
    logits = self.model.compute_logits(hidden_states)
    sample_output = self.sampler(logits)
    
    return sample_output
```

---

## 3. Data Flow: Request to Token

1.  **Request Arrival**: `AsyncLLM.add_request` pushes a request to the Scheduler.
2.  **Engine Tick**: `EngineCore.step()` triggers.
3.  **Scheduling**: [Scheduler](file:///home/frank/dev/mini-vllm/vllm/vllm/v1/core/sched/scheduler.py#67-2281) looks at available KV cache blocks and decides to run Request A (Prefill) and Request B (Decode).
4.  **Hardware Dispatch**: [Executor](file:///home/frank/dev/mini-vllm/vllm/vllm/v1/executor/abstract.py#36-355) sends the `SchedulerOutput` to the GPU Worker.
5.  **Preparation**: [GPUModelRunner](file:///home/frank/dev/mini-vllm/vllm/vllm/v1/worker/gpu_model_runner.py#386-6722) looks up where Request A and B's KV caches are stored on the GPU. It creates a batch tensor.
6.  **GPU Forward**: The model runs on the batch.
7.  **Sampling**: The next token for each request is sampled from the logits.
8.  **Feedback**: The engine receives the new tokens, updates the Scheduler, and the cycle repeats.

## 4. Key Takeaways for Minimal Implementation

To build a "mini-vLLM," you need to implement a simplified version of these four stages:
1.  **Simple Request State**: A class to track `prompt_ids`, `generated_ids`, and `current_pos`.
2.  **Batcher**: A scheduler that takes N requests from a queue and forms a batch.
3.  **Model Wrapper**: A class that takes your [GPT](file:///home/frank/dev/mini-vllm/litgpt/litgpt/model.py#22-271) model and handles the `idx` and `input_pos` logic.
4.  **KV Cache Buffer**: A pre-allocated tensor that you manually index or a simplified `PagedAttention` implementation if you want to support non-contiguous caches.
