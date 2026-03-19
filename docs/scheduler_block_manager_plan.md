# Scheduler and Block Manager Integration Plan

This document updates the scheduler/block-manager plan using two inputs:

- the current `mini-vllm` code layout in `src/engine`, `src/worker`, and `src/request.py`
- nano-vllm's actual engine split (`scheduler.py`, `block_manager.py`, `llm_engine.py`)

The goal is not to copy nano-vllm verbatim. The goal is to adopt the parts that fit this repo's existing architecture and avoid wiring engine-side admission logic into worker-side cache code.

## 1. Recommendation

The best fit for this project is:

- keep the **engine async**, so requests can be added while the engine is already serving other requests
- keep the **Scheduler** in `src/engine` as the owner of request queues, batching, and post-step request state transitions
- add a small **engine-side allocation layer** in `src/engine/block_manager.py`
- keep the existing **cache manager** in `src/worker/cache_manager.py` focused on KV plan sizing, GPU tensor allocation, and worker-side input preparation

In short:

- **Async engine loop** keeps serving while new requests arrive
- **Scheduler** decides what runs
- **Block manager / allocation manager** decides whether it fits
- **Cache manager** owns tensors and prepares model execution inputs

This is closer to nano-vllm than the current plan's "either/or" framing, and it fits this repo better than moving allocation into `CacheManager`.

## 2. What Nano-vLLM Actually Does

Nano-vllm uses a very clean engine loop:

1. `llm_engine.step()` asks the scheduler for the next batch.
2. The scheduler returns the scheduled sequences plus a prefill/decode flag.
3. The model runner executes that batch.
4. The scheduler postprocesses sampled tokens and frees finished sequences.

Just as importantly, new requests can arrive while that loop is already running. The scheduler does not build one static batch and wait for it to finish; it continuously rebuilds the next step from the latest `waiting` and `running` sets.

Its scheduler owns:

- `waiting` and `running` queues
- prefill-first admission
- decode scheduling
- recompute-style preemption
- postprocess of generated tokens

Its block manager owns:

- the free block pool
- block allocation and deallocation
- append-time block growth
- prefix-cache reuse via block hashes

Two important takeaways from nano-vllm:

- the scheduler talks to a memory controller, not to GPU tensors directly
- the memory controller is still part of the engine-side control plane; it is not the model runner

That second point matters a lot for this repo.

## 3. Current Mini-vLLM Constraints

The current codebase already has a useful split:

- `AsyncEngine` is the top-level orchestrator
- `ModelRunner` owns model loading and execution
- `BaseCacheManager` already handles KV sizing, KV allocation, and input preparation
- `Request` already mirrors nano-vllm's `Sequence` closely enough to reuse the same scheduling model

The important boundaries today are:

- `src/engine/async_engine.py` initializes the KV plan before the scheduler exists
- `src/worker/cache_manager.py` is effectively worker-side support code used by `ModelRunner`
- `src/worker/interface.py` already defines a `SchedulerOutput`
- `src/request.py` already exposes `num_cached_tokens` and `block_table`

That means the repo is already halfway to a good design:

- the top-level engine abstraction already points in the right direction for async serving
- request lifecycle belongs in the engine
- physical KV tensors belong in the runner/cache-manager path

What is still missing is the middle layer that converts "I want to run this request" into "that request can/cannot consume more KV capacity."

## 4. Why The Block Manager Should Stay Out Of CacheManager

The current plan listed two equivalent choices: separate block manager vs block logic inside the cache manager. For this repo, they are not equivalent.

Putting admission/allocation inside `CacheManager` would blur two layers that are currently cleanly separated:

- `Scheduler` runs before model execution and should be able to decide admission without reaching into runner internals
- `CacheManager` is already the execution-side component that owns tensors and attention metadata

If `Scheduler` had to call into `CacheManager.allocate_slots(...)` directly, we would be mixing:

- engine policy
- worker execution state
- backend-specific tensor concerns

That would make the scheduler harder to keep backend-agnostic and would couple future queueing logic to model-runner details.

So the recommended split is:

- `CacheManager` owns **GPU memory objects**
- `BlockManager` owns **logical capacity accounting**

The only thing the block manager needs from KV planning is metadata such as:

- `num_gpu_blocks`
- `max_seq_length`
- `BLOCK_SIZE`
- backend type (`standard` or `paged`)

It does not need the actual key/value tensors.

## 5. Recommended Component Boundaries

### 5.1 AsyncEngine

`AsyncEngine` should keep doing initialization first:

1. load model
2. build KV plan
3. allocate KV tensors
4. construct the scheduler-side allocator
5. construct the scheduler

Concretely, after `_initialize_kv_caches()` it should create:

- `PagedBlockManager(num_gpu_blocks, BLOCK_SIZE, max_seq_length)` when `kv_cache_manager == "paged"`
- a trivial dense allocator when `kv_cache_manager == "standard"`

Then `AsyncEngine` becomes the owner of an async serving loop:

1. accept new requests at any time through `add_request(...)`
2. enqueue them into the scheduler without stopping the engine loop
3. call `scheduler.schedule()` for the next step using the latest queue state
4. pass `SchedulerOutput` to `model_runner.execute_model(...)`
5. call `scheduler.postprocess(...)` with sampled tokens
6. repeat immediately so newly arrived requests can join future steps

This is the core of continuous batching: the batch is recomputed every step, not fixed for the lifetime of a request.

### 5.2 Scheduler

`src/engine/scheduler.py` should become the engine brain.

It should own:

- `waiting` and `running` queues
- `add_request(...)`
- `schedule()`
- `postprocess(...)`
- optional `abort_request(...)`

It should not own:

- block free lists
- GPU KV tensors
- attention metadata construction beyond selecting `block_tables`

The scheduler algorithm should be nano-vllm-like, but adapted to this repo's batch format:

1. try to admit waiting requests for prefill, bounded by `max_num_seqs`, `max_num_batched_tokens`, and allocator capacity
2. if at least one prefill request is admitted, return a prefill batch
3. otherwise schedule decode for currently running requests
4. after model execution, append sampled tokens, mark finished requests, and free their capacity

For the first implementation, prefer nano-vllm's simple behavior:

- full-prompt prefill, not chunked prefill
- recompute-only preemption at most
- no CPU swap
- no prefix cache requirement

That gives the project a minimal end-to-end scheduler that matches the current code maturity.

For async serving, `add_request(...)` should be safe to call while requests are already in `running`. In practice that means:

- new requests are appended to `waiting` immediately
- they are considered at the next scheduling tick
- no in-flight batch needs to be rebuilt mid-forward-pass

That keeps the concurrency model simple while still giving true continuous batching.

### 5.3 Block Manager / Allocation Layer

`src/engine/block_manager.py` should expose a small allocator interface used by the scheduler.

Recommended interface:

```python
class AllocationManager(Protocol):
    def can_allocate(self, request: Request) -> bool: ...
    def allocate(self, request: Request) -> None: ...
    def can_append(self, request: Request) -> bool: ...
    def append(self, request: Request) -> None: ...
    def free(self, request: Request) -> None: ...
```

The file can then contain two implementations:

- `PagedBlockManager`
- `DenseSlotManager`

That keeps the scheduler backend-agnostic while still letting the paged path use real block tables.

#### `PagedBlockManager`

This should be the nano-vllm-inspired implementation.

Phase 1 responsibilities:

- maintain a free list of block IDs
- assign block IDs into `request.block_table`
- grow the table when decode crosses a block boundary
- free blocks when a request finishes or is preempted

Phase 1 should be intentionally simpler than nano-vllm:

- do not require hash-based prefix caching yet
- do not require shared-reference blocks yet
- do not require copy-on-write behavior yet

Those are good later extensions, but they are not needed to integrate with the current architecture.

#### `DenseSlotManager`

The dense backend still needs scheduler-visible capacity accounting.

It can be much simpler:

- track free sequence slots, not shared blocks
- bind each running request to a slot ID
- reject admission when no slot is free
- reject append when `request.num_tokens >= max_seq_length`

The scheduler can use the same loop for both backends; only the allocator changes.

### 5.4 CacheManager

`src/worker/cache_manager.py` should remain focused on:

- `build_kv_cache_plan(...)`
- `initialize_kv_cache(...)`
- `update_states(...)`
- `prepare_model_inputs(...)`
- `forward(...)`

When paged execution is added, the cache manager should consume scheduler output such as:

- `block_tables`
- request ordering
- token positions

and convert that into the paged-attention metadata the model needs.

That is a better fit than asking the scheduler to compute physical slot mappings itself.

## 6. Request And Output Contracts

### 6.1 Request

`Request` is already close to what the scheduler needs:

- `token_ids`
- `num_prompt_tokens`
- `num_tokens`
- `num_cached_tokens`
- `block_table`
- `status`

This is enough for a nano-vllm-style minimal scheduler.

One useful invariant to keep:

- `num_cached_tokens` means "tokens whose KV is already materialized or logically reused"

That makes prefill and decode slicing straightforward.

### 6.2 SchedulerOutput

The current `SchedulerOutput` is missing one important thing: request identity.

The recommended shape is:

```python
@dataclass
class SchedulerOutput:
    requests: list[Request]
    input_ids: list[list[int]]
    positions: list[list[int]]
    block_tables: list[list[int]]
    slot_mappings: list[list[int]]
```

Why add `requests`:

- `ModelRunnerOutput.sampled_token_ids` is returned in batch order
- the engine or scheduler needs a stable mapping back to the original requests
- future paged execution will also need request order when syncing state

For an async engine, this batch object is also the boundary between:

- mutable scheduler queues
- one immutable execution step currently being run by the model

That separation makes it safe to accept new requests while the current step is executing.

The current parallel-list design for `input_ids`, `positions`, and `block_tables` is still fine.

### 6.3 How Scheduler Builds Batch Inputs

For this repo, the scheduler should output actual token slices, not just "number of scheduled tokens."

Recommended rules:

- prefill request:
  - `input_ids = request.token_ids[request.num_cached_tokens:request.num_tokens]`
  - `positions = list(range(request.num_cached_tokens, request.num_tokens))`
- decode request:
  - `input_ids = [request.last_token]`
  - `positions = [request.num_tokens - 1]`

This matches the current `ModelRunner.execute_model(...)` contract and keeps input preparation simple.

## 7. Step Lifecycle

Recommended end-to-end flow:

1. `AsyncEngine.add_request(...)` creates a `Request` and pushes it into `Scheduler.waiting`.
2. The async engine loop keeps running even as new requests arrive.
3. Each engine tick calls `scheduler.schedule()`.
4. `Scheduler.schedule()` selects requests and consults the allocator:
  - `allocate(...)` for first admission
  - `append(...)` when decode needs a new block or slot capacity check
5. `Scheduler.schedule()` returns `SchedulerOutput`.
6. `ModelRunner.execute_model(...)` runs the batch.
7. `scheduler.postprocess(...)`:
  - appends sampled tokens
  - advances request status
  - frees finished requests through the allocator
8. the next tick may admit newly arrived requests into the batch
9. finished outputs are returned by the engine

This mirrors nano-vllm's control flow, but uses this repo's richer batch payload.

The key async property is that admission is step-granular:

- requests can arrive at any time
- they join on the next scheduling step
- the engine continuously mixes old decode work with newly admitted prefill work

## 8. Preemption Strategy

Nano-vllm already includes recompute-style preemption:

- if decode cannot append, it evicts another running sequence
- the evicted sequence is moved back to waiting
- its blocks are freed
- later it is prefetched again

That model can fit here, but it should be a second step, not the first one shipped.

Recommended staging:

- Phase 1:
  - no prefix caching
  - no swap
  - optional no-preemption path for the first working scheduler
- Phase 2:
  - add recompute preemption for paged allocation pressure
- Phase 3:
  - add chunked prefill
  - add prefix caching
  - consider swap/offload only if needed

This order keeps the plan aligned with the current implementation state: the runner can already execute dense batches, but paged execution and advanced cache reuse are not wired yet.

## 9. Concrete Module Plan

### `src/engine/scheduler.py`

Implement:

- queue ownership
- scheduling loop
- request-to-batch conversion
- postprocess

Constructor inputs should include:

- `VllmConfig`
- allocator instance
- `max_seq_length`
- `eos`

### `src/engine/block_manager.py`

Implement:

- `AllocationManager` protocol
- `PagedBlockManager`
- `DenseSlotManager`

This module should mutate only request allocation fields such as:

- `block_table`
- `num_cached_tokens`
- dense `slot_id` if you add one later

### `src/engine/async_engine.py`

Extend it to:

- expose async-safe request submission
- create the allocator after KV planning
- inject the allocator into `Scheduler`
- own the async `schedule -> execute_model -> postprocess` loop
- continuously wake and schedule as long as there are pending or running requests

### `src/worker/interface.py`

Extend `SchedulerOutput` to carry:

- `requests`
- existing per-request token/position payloads
- backend-specific mapping fields as needed later

## 10. Final Call

The best integration is:

- **async engine with step-wise continuous batching**
- **nano-vllm-style scheduler ownership**
- **engine-side allocator/block-manager abstraction**
- **existing cache manager kept as worker-side execution support**

That gives this repo:

- request admission while serving is already in progress
- clean separation of concerns
- one scheduler loop for both standard and paged backends
- room to add prefix caching later without rewriting the engine boundary

Most importantly, it matches the current codebase better than moving block allocation into `CacheManager`, because the current project already has a strong engine/runner split and should preserve it.