# Experimental tt-mlir patches (NOT auto-applied)

Patches here are **opt-in experiments** — `scripts/apply-tt-mlir-patches.sh`
globs only the top-level `*.patch`, so nothing in this subdir is applied to the
default build. They preserve experiments that are intentionally *not* on the
`mola-local` branch / `tt/dev` (so the fork and the default patch series stay
consistent). Apply manually with `git -C third_party/tt-mlir apply <file>` to
reproduce.

## 23-cb-depth-knob-opt3-2026-06-08.patch — opt #3, deeper software pipelining

`lib/Dialect/D2M/Transforms/MarkSynchronizedBuffers.cpp`. Adds `MOLA_TT_CB_DEPTH=N`
to override the CB prefetch depth (`numStreamBuffers`, default 2 = double-
buffered) so the reader prefetches N-1 tiles ahead. **NEGATIVE result on TT**
(see `progress/tt-opt-pipeline-depth.md`): no wall-clock gain on dispatch-bound
shapes, and depth 4/8 overflow L1 on compute-substantial ones — depth-2 is the TT
sweet spot. Default (no env) is identical to upstream, so the patch is harmless
if applied, but it is kept here (not in the default series) because the
optimization's payoff is architecturally **NVGPU** (cp.async + SMEM headroom),
which is the forward goal.
