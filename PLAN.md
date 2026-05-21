# Multi-Program Golden Accumulation — Plan

## Goal

Extend chisel's golden accumulation so the golden chain survives across
program executions. When program A produces a runtime tensor that becomes
an input to program B, B's first op should consume A's accumulated
golden (not re-seed from the device tensor).

This is **strictly an upgrade to accumulation mode**. When
`checks_config.accumulation` is off, the feature is a no-op.

## Current state

Program-level accumulation already exists:

- `ProgramState._golden_tensor_pool: Dict[SSAName, GoldenMapTensor]`
  (context.py:282) persists across ops within one program invocation.
- `_default_pre_op` (callbacks.py) seeds function args from the device
  tensor (`golden_promoted` record).
- `execute_golden_from_pool` (executor.py:116) reads inputs from the
  pool and writes outputs back.
- Pool dies with `ProgramState` at `postprogram`.

This breaks across program boundaries because `SSAName` is IR-scoped
(same `%0` collides across programs, not stable across reruns).

## Key insight: Tensor.globalId is the cross-program identity — but only via the stored Tensor

`Tensor::getGlobalId()` (types.h:413) is a uint64 atomic assigned
on Tensor construction. Verified properties:

- `submit()` returns its outputs via `gatherOutputTensors()`
  (types.cpp:197), which returns the **stored** `tt::runtime::Tensor`
  objects from `liveTensors`. Those Tensors carry the globalIds
  assigned when their producing op ran (`insertTTNNTensorAndValidate`).
- When the user passes those same Tensor objects as inputs to a
  subsequent `submit()`, they are inserted into the new pool's
  `liveTensors` **with their globalIds preserved**.
- Therefore, the globalId is a stable cross-program identifier — as
  long as we read it from the **stored** Tensor, not a freshly-wrapped
  one.

**Critical pitfall — verified during planning:** the existing
`retrieveTensorFromPool` (runtime.cpp:2186) does NOT return the stored
Tensor. It calls `createRuntimeTensorFromTTNN(...)` to build a fresh
wrapper (new globalId) and then `toHost()` (untilize/copy). Using it
to read the cross-program globalId is broken. We need a separate
accessor — see step 1 below.

(Note: `TensorRef::global_id` is a flatbuffer slot index — stable per
binary but NOT the right key here.)

## Design

Add a session-scoped pool keyed by `Tensor.globalId`, populated at
program outputs and consulted at program inputs. Eviction via
`registerOnDestroyCallback` on the underlying TTNN tensor wrapper.

```
ChiselContext
  _program_io_pool: Dict[int, GoldenMapTensor]   # NEW (session-scoped)
  binaries
    BinaryState
      programs
        ProgramState
          _golden_tensor_pool: Dict[SSAName, GoldenMapTensor]  # existing (program-scoped)
```

### Flow

- **Program A POST-op** for a function-output SSA: get the **stored**
  device tensor (new runtime API), read its `globalId`, store the
  accumulated golden into `_program_io_pool[global_id]`. Register a
  destroy callback that evicts the entry when the underlying tensor
  wrapper is destroyed.
- **Program B PRE-op** for a function-arg SSA: get the **stored**
  device tensor, read its `globalId`, look it up in
  `_program_io_pool`. Hit → seed
  `ProgramState._golden_tensor_pool[ssa]` from the cross-program
  golden. Miss → existing `golden_promoted` (seed from device).

## Implementation steps

### 1. Split `retrieveTensorFromPool` into device-tensor and host-copy accessors

`runtime/lib/ttnn/runtime.cpp:2186` currently couples two concerns:
fetching the stored Tensor and host-copying it. Split into:

- **`getDeviceTensorFromPool(CallbackContext, TensorRef) -> Tensor`** —
  returns `tensorPool.getRuntimeTensorAndValidate(tensorRefPtr)` (by
  value, preserving globalId). No host transfer. This is what chisel
  needs to read the stable globalId.
- **`retrieveTensorFromPool(...)`** — keep existing host-copy
  semantics for callers that actually need host data.

Plumb the new function through:

- `runtime/include/tt/runtime/runtime.h` — declaration
- `runtime/lib/runtime.cpp` — dispatch
- `runtime/include/tt/runtime/detail/ttnn/ttnn.h` + impl
- `runtime/include/tt/runtime/detail/ttmetal/ttmetal.h` + impl (or
  `LOG_FATAL` not-implemented stub for now)
- `runtime/python/runtime/runtime.cpp` — nanobind binding
  `get_device_tensor_from_pool`
- Also add `.def("get_global_id", ...)` to the `Tensor` nanobind
  class (still needed; today there's no Python access to it).

### 2. Add session-scoped pool on `ChiselContext`

`tools/chisel/chisel/context.py:44` — in `ChiselContext.__init__`:

```python
self._program_io_pool: Dict[int, GoldenMapTensor] = {}
```

Add a property `program_io_pool` returning it (mirrors the existing
`golden_tensor_pool` property).

Scope: session (bind→unbind). Cleared on `unbind()`. Not per-binary —
tensors can cross binary boundaries (split graphs).

Eviction is event-driven via destroy callbacks (step 5); the
unbind-time clear is a safety net.

### 3. Pre-compute function-arg / function-return SSA sets per program

`tools/chisel/chisel/context.py:260` — in `ProgramState.__init__`:

```python
asm_state = ir_module.get_asm_state()
self._function_arg_ssas: Set[SSAName] = {
    arg.get_name(asm_state)
    for arg in ir_module.get_function_inputs(program_name)
}
self._function_output_ssas: Set[SSAName] = {
    out.get_name(asm_state)
    for out in ir_module.get_function_outputs(program_name)
}
```

Cached once per program to avoid per-op walks. `IRModule` already
exposes both lists (ops.py:125, 129).

### 4. PRE-op: function-arg lookup in cross-program pool

`tools/chisel/chisel/callbacks.py` `_default_pre_op` — for an input
SSA not yet in `pool`. Gated on `checks_config.accumulation`; if
accumulation is off, take the existing device-seed path only.

- If `ssa in function_arg_ssas`:
  - Fetch device tensor via the new `get_device_tensor_from_pool`.
  - Read `global_id`. Look up in `ctx.program_io_pool`.
  - Hit: `pool[ssa] = cross_pool[global_id]`. Record
    `golden_promoted` with `source="program_pool"`.
  - Miss: existing path. Record `golden_promoted` with
    `source="device"`.
- If `ssa not in function_arg_ssas` and not in pool: emit a warning
  record and fall back to seed-from-device. (Per discussion: be
  transparent in the report but keep the program running, don't
  raise.)

### 5. POST-op: publish function outputs to cross-program pool + register destroy callback

`tools/chisel/chisel/callbacks.py` `_default_post_op` — for each
output where `ssa in function_output_ssas`:

- **Skip entirely if `checks_config.accumulation` is off.** This
  feature only applies when accumulation is on; otherwise there is
  no accumulated golden worth publishing.
- Otherwise:
  - Fetch the stored device tensor via
    `get_device_tensor_from_pool`. Read `global_id`.
  - Write `ctx.program_io_pool[global_id] = accum_out`.
  - Register an on-destroy callback on the underlying tensor wrapper
    that evicts `global_id` from `program_io_pool`. (See "Eviction"
    below.)

**Which golden to publish:** accumulated, not isolated.

- Isolated uses device-derived inputs (already "promoted"
  mid-program) — publishing it would defeat the purpose of a
  cross-program chain.

### 6. Eviction via `registerOnDestroyCallback`

`TTNNTensorWrapper::registerOnDestroyCallback` (types.h:69) fires
when the underlying TTNN tensor wrapper is destroyed (last reference
released). We exploit this for automatic pool eviction:

- Expose a thin Python binding that, given a runtime `Tensor`, walks
  to its `TTNNTensorWrapper` handle and registers a callback. The
  callback captures the `global_id` (uint64, plain value — no Python
  refs into the destructor) and a weakref/handle to the
  `ChiselContext`'s `_program_io_pool`.
- On destroy: pop `global_id` from `_program_io_pool` if present. The
  callback runs on the runtime thread that destroys the wrapper —
  guard the pool mutation appropriately if chisel ever runs
  multi-threaded (today it doesn't; a note is enough).
- Safety net: `unbind()` still clears the pool unconditionally to
  cover any tensors that outlive the session.

This bounds memory: pool entries die with their tensors, no eviction
policy or LRU needed.

### 7. Report payload extension

`tools/chisel/chisel/report.py` — extend `GoldenPromotedPayload`:

```python
@dataclass
class GoldenPromotedPayload:
    source: Literal["device", "program_pool"] = "device"
```

Backward compatible (existing JSON consumers see the new field as
defaulted). Avoids inventing a new record kind for a closely related
event.

Also emit a distinct warning record kind for the "non-arg SSA missing
from pool" case in step 4 — keep it visible in the report rather than
silent.

## Files touched

- `runtime/include/tt/runtime/runtime.h` — new accessor decl
- `runtime/lib/runtime.cpp` — dispatch
- `runtime/include/tt/runtime/detail/ttnn/ttnn.h` + impl —
  `getDeviceTensorFromPool`
- `runtime/include/tt/runtime/detail/ttmetal/ttmetal.h` + impl — stub
- `runtime/python/runtime/runtime.cpp` — `get_device_tensor_from_pool`,
  `Tensor.get_global_id`, on-destroy callback binding
- `tools/chisel/chisel/context.py` — session pool + per-program SSA
  sets
- `tools/chisel/chisel/callbacks.py` — pre/post lookups + publishes +
  destroy-callback registration
- `tools/chisel/chisel/report.py` — payload field + warning record

## Tests (separate PR)

Per discussion: testing goes in a follow-up PR with broader scenarios.
Plan two complementary tests:

1. **Two distinct programs in one session.** Program A returns a
   tensor that is fed as input to program B. Assert:
   - chisel report contains `golden_promoted` on B's first op with
     `source="program_pool"`
   - `_program_io_pool` is non-empty between the two `submit` calls
   - accumulated PCC on B remains within tolerance
2. **Same program executed two/three times with re-used tensors.**
   The second and subsequent runs should also see
   `source="program_pool"` seeding on their function args.

Both tests live under `test/python/chisel/` and use
`_ttmlir_runtime` directly (not `ttrt` — per project convention).

## Out of scope (deliberately)

- Publishing every intermediate op's output to the cross-program
  pool. Only function outputs go in — matches the
  "inputs/outputs of the program" framing and keeps memory bounded.
- Cross-process / cross-session identity. globalId is process-local;
  not a concern for chisel's single-session model.
- Concurrency. Chisel runs serially today; the destroy-callback path
  is the only one that touches the pool from a non-main context, and
  a single lock can be added if needed later.
- Per-binary scoping. Session-wide is simpler and correct for
  cross-binary flows; tighten later if it causes issues.

## Assumption verification log

- **Verified (planning phase):** `tt::runtime::Tensor::globalId` is
  preserved across program boundaries when the **stored** Tensor is
  accessed. `gatherOutputTensors` (types.cpp:197) returns stored
  tensors by value, preserving globalId; the same objects pass into
  the next program's `liveTensors` intact.
- **Verified (planning phase):** the existing
  `retrieveTensorFromPool` returns a freshly-constructed Tensor and
  cannot be used to read the stable globalId. Hence step 1.
- **Verified (planning phase):** `registerOnDestroyCallback`
  (types.h:69) exists on `TTNNTensorWrapper` and is suitable for
  eviction.
