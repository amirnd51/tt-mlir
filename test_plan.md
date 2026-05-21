# Chisel Accumulation Mode — Golden Promotion Test Plan

Scenario inventory for accumulation-mode golden promotions, grouped by what each
group is actually exercising. Existing coverage in
`test/python/chisel/test_multi_program_accumulation.py` is flagged with ✓.

## A. Baselines (sanity)
1. **accumulation=False**: no `PROGRAM_POOL` records ever; `program_io_pool` stays empty across A→B; no destroy callbacks registered.
2. **Single-program, accumulation=True**: function args all get `DEVICE`-source promotion records; no `PROGRAM_POOL` (nothing to chain from yet).
3. **First op of B**: exactly one promotion record per function arg, no intermediate-SSA promotions (a producer missing a golden would surface here).

## B. Multi-program chaining (the happy path)
4. ✓ A→B with one chained input.
5. ✓ A→A→A re-running the same binary; runs 2 and 3 each contribute ≥1 `PROGRAM_POOL` promotion.
6. **A→B→C** linear chain across three distinct binaries.
7. **Diamond fan-out**: A's output consumed by both B and C in any order. Two `PROGRAM_POOL` lookups against the same `globalId`.
8. **Partial chain**: only some of B's inputs come from A; rest are fresh host tensors. Mixed `PROGRAM_POOL` / `DEVICE` records on B's first op.
9. **Multi-output A**: distinct A outputs feed different B inputs.
10. **Identity program**: A's function arg is also a function output → it must still be published to the cross-pool.

## C. Pool lifetime / eviction
11. **Drop A's output before B's submit** (`del out_a`) → destroy callback evicts; B falls back to `DEVICE`.
12. **Explicit `ttnn.deallocate` of A's output** between submits → eviction.
13. **Session boundary**: `unbind()` (i.e. exiting `chisel.session`) empties `program_io_pool`, even if device tensors are still live.
14. **Two sequential sessions**: no leakage from session 1's pool into session 2.
15. **Long chain (10+ programs)**: pool size stays bounded to currently-live tensors; old `globalId`s evicted.

## D. Identity / layout preservation (the chain is keyed by `globalId`)
16. **`to_layout` between programs creates a new tensor** → globalId changes → chain falls back to `DEVICE`. Documents what breaks the chain.
17. **Pass-through verbatim** (current test pattern) → globalId preserved → chain works.
18. **Mixed: chained slot 0 verbatim, slot 1 newly laid-out** (current test pattern). Both flows must coexist on the same submit.
19. **Layout mismatch between A's output and B's expected input layout** → shape/dtype check on B's first op fails cleanly, no crash.

## E. `func.CallOp` (sub-program via call)
20. **Single call to a helper func**: parent op's `_subprogram_pre_op` publishes inputs to cross-pool by globalId; callee's PRE sees `PROGRAM_POOL` source on its function args.
21. **Callee output round-trips into parent's pool**: `_subprogram_post_op` installs the golden at the parent SSA; subsequent parent ops consume it (`ACCUMULATED` numerics record, not just `ISOLATED`).
22. **Nested calls**: helper A calls helper B. Two levels of cross-pool publish/lookup; chain must hold end to end.
23. **func.call with a parent input that's `no_golden` upstream**: `pool.get(parent_ssa)` returns None, publish skipped, callee's first op falls back to `DEVICE` for that arg only. The other args still get `PROGRAM_POOL`.
24. **func.call with multiple outputs**: each output independently published and looked up.
25. **func.call followed by more parent ops using its result**: end-to-end accumulated chain through the call.

## F. `ttcore.LoadCachedOp`
26. **Cold path (cache miss)**: behaves like func.call — sub-program executes, goldens computed, published, parent installs them.
27. **Warm path (cache hit)**: sub-program callbacks do NOT fire. `_subprogram_post_op` finds no entry in `program_io_pool` → promotion source = `DEVICE`, chain restarts at the cached output. Verify a record is emitted with source=DEVICE and no numerics check (since `golden is None`).
28. **LoadCachedOp inside an A→B accumulation chain**: cache hit doesn't break downstream chaining — chain just re-seeds at the cached boundary.
29. **Same LoadCachedOp hit on first submit of a fresh session** (the comment in `_subprogram_post_op` calls this out): explicit test that this degrades gracefully and emits the right record.

## G. `DeallocateOp` interaction
30. Deallocate of an SSA inside the program emits `golden_evicted`; subsequent attempts to use that SSA (if any) surface as bugs in the IR, not chisel crashes.
31. Deallocate of a function output before B consumes it (rare/forbidden, but) — pool entry should still be present if globalId is alive (publish is at POST of A's last op).

## H. Failure / degradation paths
32. **Op with `no_golden` produces a function output**: that output is never published to the cross-pool; B's first op for that input gets `DEVICE`, not `PROGRAM_POOL`.
33. **PCC failure in A**: A's accumulated golden is still published (the chain uses the golden, not the device). Document whether `NUMERICS_FAIL` in A blocks publishing or not — currently it does not, since publish is unconditional when `accum_out is not None`.
34. **PRE failure on an op in A** → `pre_failed=True`, POST skipped, no publish for that op's outputs. If that op was the function output, B sees `DEVICE`.
35. **Golden produces NaN/inf**: pool stores it; B's PCC may report fail but no crash.
36. **Sub-program PRE handler exception**: chisel_safe records `chisel_bug`; sub-program callee still runs and seeds from device.

## I. Multi-shard / multi-device
37. Multi-mesh tensors crossing program boundaries — lookup-by-`globalId` should still work; per-shard golden parity preserved.
38. Programs with `MeshShardOp` / `MeshPartitionOp` (no_golden) at the boundary — chain breaks at that op; document.

## J. Trace ops in the middle of a chain
39. `BeginTraceCaptureOp` / `ExecuteTraceOp` / `CaptureOrExecuteTraceOp` in A: no_golden, but A's pre-trace ops still publish goldens normally; chain continues into B.
40. **Trace replay path**: first submit captures+executes (callbacks fire), subsequent submits replay-only — does chisel see ops on replay? If not, chain semantics on the second submit need a documented expectation.

## K. Audit invariants (assert in every multi-program test)
41. After A→B: `count(promotions on B.first_op with source=PROGRAM_POOL) == count(B function args carried over from A)`.
42. Every function-arg SSA gets exactly one promotion per program submit; intermediate SSAs get zero promotions.
43. JSONL output: each record has correct `binary_id` and `program_name`; A's and B's records are distinguishable.
44. After `unbind()`: `ctx.program_io_pool == {}` regardless of whether device tensors are still live.

## L. Cross-cutting "weird shapes"
45. Program with **zero function args** (all-constant) → no promotions at all.
46. Program with **zero function outputs** (side-effects only, if reachable) → nothing published.
47. **Reusing the same host tensor as both inputs to A** (`x=y` aliasing) → both function-arg SSAs share a single entry by globalId; one publish, two lookups.

---

## Priority

Not-yet-exercised scenarios to prioritize:
- **D16** — `to_layout` breaks the chain (explicit, documented).
- **C11 / C13** — eviction via destroy callback, and unbind clears the pool.
- **E20–E25** — func.call across the board.
- **F27 / F29** — LoadCachedOp cache-hit degradation.
- **K41** — count invariant on audit records.
