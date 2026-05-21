# Plan — `func.call` and `ttcore.LoadCachedOp` via globalId pool

## Background — why the globalId pool already does most of the work

The existing multi-program implementation (`callbacks.py:239 _publish_program_output`,
`callbacks.py:124 cross-program lookup`) chains accumulated goldens across program
boundaries by:

- POST-op: when an output SSA is a function-output of the program, publish its
  accumulated golden into `ChiselContext._program_io_pool[tensor.globalId]` and
  register a destroy callback for eviction.
- PRE-op: when an input SSA is a function-arg of the program, look up
  `tensor.globalId` in `_program_io_pool` and seed the program-scoped
  `golden_tensor_pool`.

Runtime facts that make this directly reusable for sub-programs:

1. `ProgramExecutor::execute()` (program_executor.cpp:191) fires **both**
   `preProgramCallback`/`postProgramCallback` **and** the per-op `pre_op`/`post_op`
   hooks for *any* invocation — including the nested executor that `func.call`
   (func_call.cpp:23) and `LoadCachedOp` cache-miss (load_cached.cpp:70) construct.
   So chisel's `preprogram`/`postprogram`/`pre_op`/`post_op` already fire for
   sub-programs; a fresh `ProgramState` is created and its `function_arg_ssas` /
   `function_output_ssas` are pre-computed (context.py:314).
2. Sub-program **inputs** are pulled from the parent's
   `tensorPool.getRuntimeTensorAndValidate(...)` (load_cached.cpp:64, func_call.cpp:18)
   — the **same** `runtime::Tensor` objects the parent has, so `globalId` is stable
   across the boundary.
3. Sub-program **outputs** come from `gatherOutputTensors()` (types.cpp:197) —
   returns stored Tensors by value, preserving the globalIds assigned during the
   sub-program's execution — and are inserted into the parent's pool with those
   globalIds (load_cached.cpp:86, func_call.cpp:33).
4. **`LoadCachedOp` cache hit** (load_cached.cpp:46): the sub-program does **not**
   run, but the cached Tensors inserted into the parent pool keep their **original**
   globalIds (from the cache-miss execution that created the cache entry). Since
   those tensors were `setRetain(true)` (line 80), their `TTNNTensorWrapper`s don't
   get destroyed, so the destroy-callback evictor never fires and the
   `_program_io_pool` entry survives across the cache lifetime *within a session*.

Implication: the *only* gap in reusing the existing multi-program logic is that
the existing `_default_pre_op` publish step happens for **function outputs of the
program currently executing**, not for **inputs to a `func.call` / `LoadCachedOp`**
mid-program. We need to bridge that one edge.

## The bridge — two custom handlers, no new state

### `_subprogram_pre_op(ctx, config)` — for `func.CallOp` and `ttcore.LoadCachedOp`

Run only when `checks_config.accumulation` is on (otherwise no-op + standard
`no_golden` semantics).

For each `(mlir_input, rt_tensor_ref)` of the parent op:

- Stash host copy of the input (same as default pre-op, so POST shape/dtype checks
  still work).
- If `parent_ssa in ctx.golden_tensor_pool` (parent's accumulated golden for this
  SSA exists): publish it to `_program_io_pool` keyed by the input Tensor's
  globalId, plus register the destroy-callback evictor. Factor the body of
  `callbacks.py:239 _publish_program_output` into a helper so the same code is
  reused — no new state, same eviction discipline.
- If absent (e.g. upstream op was `no_golden` and never seeded a golden): skip —
  the sub-program's `_default_pre_op` will fall back to `source=device` for that
  arg, which is the correct degradation.

No isolation golden runs (these ops have no Python golden function). No
`golden_promoted` record emitted here — the sub-program emits its own
`golden_promoted` records for its function-arg SSAs, which already carries
`source=program_pool` (per `GoldenPromotionSource`).

### `_subprogram_post_op(ctx, config)` — for `func.CallOp` and `ttcore.LoadCachedOp`

Run only when accumulation is on; otherwise no-op.

After the sub-program completes (whether via real execution or cache hit), for
each `(mlir_output, output_ref)`:

- shape/dtype-validate the output ref against the MLIR output (as
  `_default_post_op` does).
- Look up the output Tensor's `globalId` via
  `tt_runtime.get_tensor_global_id_from_pool(rt_program_context, output_ref)`.
- **Hit in `_program_io_pool`**: install the found golden into the parent's
  `golden_tensor_pool[parent_output_ssa]`. Run accumulated PCC against the device
  tensor. Emit `golden_promoted` with `source=program_pool` to reflect that the
  parent SSA was seeded cross-program.
- **Miss**: (only possible for a `LoadCachedOp` cache hit in a *fresh session*
  where the original const-eval ran in a prior session) — fall back to
  seed-from-device: pull the device tensor and install it as the parent's golden
  with `source=device`. Skip accumulated PCC (no golden to compare against). This
  is the same degradation as a function-arg pool miss in current code.

The sub-program's own `_default_post_op` already publishes its function-output
goldens to `_program_io_pool` before this POST-op runs (cache miss / func.call:
just happened in the inner postprogram; cache hit: was done in a prior call
within the same session). So a **hit** is the common path and the chain is
preserved end-to-end.

No isolation PCC is emitted — the op is opaque from chisel's standpoint; the
sub-program's interior ops emit their own isolation/accumulation records.

### How this differs from the POC at 1d5652cdd9

The POC stored two extra slots on `ChiselContext` (`_pending_subprogram_seed`,
`_pending_subprogram_result`) and one extra flag on `ProgramState`
(`_is_seeded_subprogram`), plus two hooks called from `preprogram`/`postprogram`
(`seed_subprogram_pool`, `harvest_subprogram_pool`) to copy goldens by
**positional correspondence** between parent op operands and sub-function arg
SSAs.

The globalId route eliminates all of that: positional correspondence is already
enforced by the runtime (parent passes its Tensors → sub-program's tensor pool
inserts them into the slots its IR expects), and the **identity** is the
`Tensor.globalId`. The sub-program then naturally finds its golden via the
existing `_default_pre_op` cross-program lookup. No nesting-aware seed slot, no
`_is_seeded_subprogram`, no positional zips at the chisel level — and nested
calls just work because each level's pre-op publishes its own inputs by globalId,
and each level's post-op reads its own outputs by globalId, with no mutual state.

## Implementation steps

### 1. Refactor the publish helper

`callbacks.py:239 _publish_program_output` — generalize signature so it can be
called from the new pre-op handler:

```python
def _publish_to_program_pool(
    ctx: ChiselContext, output_ref: TensorRef, golden: GoldenMapTensor
) -> None: ...
```

Pure rename + drop the `program_io_pool` dict argument (read it off `ctx`
instead). Re-call from `_default_post_op` and from the new `_subprogram_pre_op`.
No behavior change.

### 2. Add a shared cross-program lookup helper

Extract the inner lookup-and-record block from `_default_pre_op` (the
`if accumulation and ssa in function_arg_ssas: ...` branch, callbacks.py:122–142)
into a tiny helper:

```python
def _seed_from_program_pool(
    ctx, ssa, rt_tensor_ref
) -> Optional[GoldenMapTensor]: ...
```

Returns the seed if hit, else `None`. Used by both the existing default pre-op
and the new sub-program post-op (for the parent's installed golden). Keeps the
`golden_promoted` record emission consistent across paths.

### 3. New handlers in `op_handlers.py`

```python
@chisel_safe
def _subprogram_pre_op(ctx, config) -> None:
    if not ctx.checks_config.accumulation:
        # Stash inputs for shape/dtype symmetry, otherwise nothing to do.
        ...
        return
    pool = ctx.golden_tensor_pool
    for mlir_input, rt_tensor_ref in zip(get_op_inputs(ctx.op), ctx.input_refs, strict=True):
        tensor = _validate_and_retrieve_tensor(ctx, mlir_input, rt_tensor_ref)
        ctx.stashed_inputs[mlir_input.get_name(ctx.asm_state)] = tensor
        parent_ssa = mlir_input.get_name(ctx.asm_state)
        golden = pool.get(parent_ssa)
        if golden is not None:
            _publish_to_program_pool(ctx, rt_tensor_ref, golden)
```

```python
@chisel_safe
def _subprogram_post_op(ctx, config) -> None:
    if not ctx.checks_config.accumulation:
        return
    asm_state = ctx.asm_state
    pool = ctx.golden_tensor_pool
    for mlir_output, output_ref in zip(get_op_outputs(ctx.op), ctx.output_refs, strict=True):
        device_tensor = _validate_and_retrieve_tensor(ctx, mlir_output, output_ref)
        ssa = mlir_output.get_name(asm_state)
        cross_gid = tt_runtime.get_tensor_global_id_from_pool(ctx.rt_program_context, output_ref)
        golden = ctx.program_io_pool.get(cross_gid) if cross_gid is not None else None
        if golden is not None:
            pool[ssa] = golden
            source = GoldenPromotionSource.PROGRAM_POOL
            check_numerics(ctx, ctx.op, ssa, golden, device_tensor, mode=NumericsMode.ACCUMULATED)
        else:
            # LoadCachedOp cache hit across sessions — chain broken, degrade to device.
            pool[ssa] = device_tensor
            source = GoldenPromotionSource.DEVICE
        ctx.write_record(ChiselRecord(op=ctx.op.name, check="golden_promoted",
                                      ssa=ssa, payload=GoldenPromotedPayload(source=source)))
```

(Both handlers reside in `op_handlers.py` to keep `callbacks.py` lean — same
locality as `_deallocate_pre_op`.)

### 4. `op_configs.py`

Replace the two `no_golden=True` entries:

```python
func.CallOp:           ChiselOpConfig(pre_op=_subprogram_pre_op, post_op=_subprogram_post_op),
ttcore.LoadCachedOp:   ChiselOpConfig(pre_op=_subprogram_pre_op, post_op=_subprogram_post_op),
```

Drop both from `get_op_names_no_golden()` semantically — they DO now produce
goldens (transitively) and should not appear in test skip-lists. If any
downstream test depends on them being in that set, update those tests.

### 5. No runtime / context changes required

- `_program_io_pool`, `get_tensor_global_id_from_pool`,
  `register_pool_tensor_destroy_callback`, `function_arg_ssas`,
  `function_output_ssas` all already exist.
- Sub-program `ProgramState` creation (preprogram fires → fresh state with
  arg/output SSA sets) already happens for nested executors.
- Cleanup via destroy callbacks already handles the bounded-memory story;
  `unbind()` clear is the safety net.

### 6. Report — no schema change

`GoldenPromotedPayload` already has `source: GoldenPromotionSource` with
`PROGRAM_POOL` and `DEVICE`. The parent's `_subprogram_post_op` reuses those
values verbatim.

## Edge cases / verification points

- **Nested sub-programs (LoadCachedOp inside func.call, or vice versa):** works
  without special handling — each level's pre-op publishes inputs by globalId,
  each level's post-op reads outputs by globalId; there is no shared state
  between nesting levels in chisel.
- **LoadCachedOp cache hit, same session:** the original execution's
  `_default_post_op` already populated `_program_io_pool` for the sub-function's
  function-output globalIds. Cached Tensors carry the same globalIds → lookup
  hits → chain continues. (Verified by reading load_cached.cpp:46–55 + the
  existing pool-survival argument from retain.)
- **LoadCachedOp cache hit, fresh session:** original execution happened in a
  prior session whose `_program_io_pool` was cleared on `unbind()`. The parent's
  `_subprogram_post_op` will miss the lookup and degrade to seed-from-device —
  chain restarts here, which is the strictly best we can do without persisting
  goldens across sessions (out of scope, same as the original plan's
  `Out of scope` for cross-process identity).
- **Sub-program ops where some inputs come from a `no_golden` op:** the parent's
  `pool.get(parent_ssa)` returns `None` → we don't publish. The sub-program's
  `_default_pre_op` then falls back to `source=device` for that arg — same as
  today's behavior at the session entry point.
- **Existing `test_multi_program_accumulation.py`:** still passes (no behavior
  change for plain top-level program-to-program flows; the new code only adds
  publish-on-sub-program-input and read-on-sub-program-output, neither of which
  is exercised by the existing two-program test).

## Files touched

- `tools/chisel/chisel/callbacks.py` — extract `_publish_to_program_pool` and
  `_seed_from_program_pool` helpers; no external API change.
- `tools/chisel/chisel/op_handlers.py` — new `_subprogram_pre_op`,
  `_subprogram_post_op` (and any small imports).
- `tools/chisel/chisel/op_configs.py` — switch `func.CallOp` and
  `ttcore.LoadCachedOp` entries off `no_golden=True` and onto the new handlers.

No changes to runtime/, context.py, report.py, or the C++ side.

## Tests (separate PR, per existing plan convention)

1. **`func.call` chain.** Top-level program calls a helper function; verify the
   helper's first op sees `source=program_pool` and accumulated PCC is within
   tolerance on the helper's outputs and back in the caller.
2. **`LoadCachedOp` cache miss → second run cache hit, same session.** First run
   populates the cache and `_program_io_pool`; second run hits the cache,
   parent's POST-op finds the golden via globalId — assert `source=program_pool`
   on the parent's outputs.
3. **`LoadCachedOp` across `unbind()`/`bind()`.** Assert graceful degradation:
   `source=device` on parent's outputs, no chisel_bug records.
4. **Nested `func.call` inside `func.call`.** End-to-end chain through two
   levels of nesting.

All tests use `_ttmlir_runtime` directly per project convention.

## Out of scope

- Persisting `_program_io_pool` across sessions (would be needed to chain
  through LoadCachedOp cache hits in fresh sessions; currently the new code
  degrades cleanly to `source=device`).
- Special handling for sub-programs that mutate inputs in place — same caveat
  as today.
