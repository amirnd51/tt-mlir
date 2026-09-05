# tt-mlir local patches

Local-only patches applied to `third_party/tt-mlir` to unblock the
DRAM-input runtime path on Blackhole. Not yet upstreamed.

> **Note (2026-06-09):** the raw `.patch` files (and the `deferred-on-bump/`,
> `experiments/`, `superseded/` subdirs) are kept **local-only** — they are a
> redundant record. The authoritative patched toolchain is the
> `amirnd51/tt-mlir` fork (`mola-local`) that the submodule gitlink points to, so
> a recursive clone gets everything without them. This README (the audit
> narrative of *what* changed and *why*) stays tracked. The patch files remain on
> disk and can be regenerated from the fork (`git format-patch`) if needed.

The submodule is pinned to the `amirnd51/tt-mlir` fork (`mola-local`), which
already carries these changes as commits, so a recursive clone gets the full
toolchain. After a fresh `git submodule update --init`, re-apply the records with:

```bash
scripts/apply-tt-mlir-patches.sh   # idempotent: skips already-applied
```

**Directory layout** (only the top level is auto-applied):
- top-level `*.patch` — the live patch set (10, 11, 12, 13, 14, 15, 16, 20, 21, 22).
- `superseded/` — obsolete (05) / superseded-by-21 (17, 19) records, NOT applied.
- `deferred-on-bump/` — the TTMetal-chain patches (01–04, 06, 08) deferred for
  re-authoring against the bumped tree; described in the list below for history.
- `experiments/` — negative-result experiments (e.g. opt #3 CB-depth knob).

## Patches

Entries 1–7 below (patches 01–04, 06, 08) currently live in `deferred-on-bump/`;
the remaining entries are the live top-level set. Active patches **11, 14, 15, 16**
are documented in the "Live patches added post-bump" subsection at the end.

1. **`01-ttir-to-d2m-branch-on-memspace.patch`** — `lib/Conversion/TTIRToD2M/TTIRToD2M.cpp`
   Branch `layoutKind` on `memSpace`: DRAM → `Interleaved`,
   anything else → `Sharded`. Was hardcoded `Sharded` for all
   memory spaces, which produced sharded-DRAM allocs that
   tt-metal's runtime can't dispatch.

2. **`02-grid-selection-dram-interleaved-bridge.patch`** — `lib/Dialect/D2M/Transforms/GridSelection.cpp`
   Two changes that together let `d2m.generic` consume raw DRAM-
   interleaved operands:
   - `hasTTNNOperands`: detect raw DRAM-interleaved operands in
     addition to the existing ttnn.cast pattern.
   - `insertTTNNDRAMStreams`: relax the cast-op-required assertion
     so the d2m.stream_layout bridge can be constructed without
     a cast op (using the operand's defining op as anchor).

3. **`03-lower-to-layout-staging-buffer-sharded.patch`** — `lib/Dialect/D2M/Transforms/LowerToLayout.cpp`
   Two changes that prevent illegal L1+Interleaved staging buffers
   when the destination is DRAM+Interleaved:
   - System→device path: force `TensorMemoryLayout::Sharded` for
     L1 staging instead of inheriting from the destination tensor.
   - `modifyDeviceType`: derive memory_layout from memory_space
     (Sharded for L1) independently of the destination's enum.

4. **`04-fullop-bufferize-element-type-cast.patch`** — `lib/Dialect/D2M/IR/D2MOps.cpp`
   `d2m::FullOp::bufferize` constructs a `DenseElementsAttr` from
   the op's `fill_value` attr (constrained to `F32Attr | I32Attr`
   by the op def) and the result tensor type. When the result has
   a narrower element type (bf16, f16, ...), `DenseElementsAttr::get`
   asserts that the FloatAttr's type matches the element type. Cast
   the fill value through APFloat with the result's float semantics
   (or APInt with bit-width) before constructing the dense attr.

5. **`05-rank-normalization-dense-resource.patch`** — OBSOLETE, moved to
   `superseded/`. Targeted `lib/Dialect/TTIR/Transforms/RankNormalization.cpp`
   (the earlier list incorrectly said `D2M`), which upstream DELETED
   (`daf610169b`). `updateConstantValueAttr` no longer exists; see
   `superseded/README.md`.

6. **`06-flatbuffer-bf16-fp16-serialization.patch`** — `lib/Target/TTMetal/TTMetalToFlatbuffer.cpp`
   `memrefGlobalOpToFlatbufferByteVector` only handled f32 (and
   asserted on every other float bit-width). Add a 16-bit branch
   that serializes the raw bit pattern via APFloat for both bf16
   and f16. Required as soon as the bf16 fixes upstream let bf16
   constants reach the flatbuffer translator.

7. **`08-ttir-constant-to-d2m.patch`** — `lib/Conversion/TTIRToD2M/TTIRToD2M.cpp`
   `D2MConstantOpRewriter` lowers `ttir.constant` (with `DenseElementsAttr`
   or `DenseResourceElementsAttr` value) to `arith.constant` +
   `d2m.to_layout`. Splat constants get canonicalized to `ttir.full`
   earlier and handled by `D2MFullOpRewriter`; this pattern handles
   the non-splat case (real model weights via `dense_resource`).
   Required for any HuggingFace Llama checkpoint flow through the new
   TTMetal chain. `arith.constant` of a tensor type is bufferizable by
   MLIR's standard bufferize patterns into `memref::GlobalOp` +
   `memref::GetGlobalOp` (which the flatbuffer translator already
   handles via `CreateHostAllocCommand`). The follow-up
   `d2m.to_layout` adds the `metal_layout` encoding.

8. **`10-getdevice-shape-pad-grid.patch`** — `lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp`
   `MetalLayoutAttr::getDeviceShape` iterates `physicalShape.size()`
   times indexing `gridShape[i]`. Higher-rank tensors (Llama 4D-6D
   activations after tile expansion) exceed the device's mesh-grid
   rank (typically 2D), so `gridShape[i]` was OOB. Pad `gridShape`
   with leading 1s to match `physicalShape.size()`. Padding 1
   semantically means "no grid distribution on that dim" which is
   correct for non-mesh dims. Required for any HuggingFace Llama
   checkpoint flow (4D batch tensors).

9. **`12-ttir-to-d2m-per-tensor-memspace-2026-06-03.patch`** — `lib/Conversion/TTIRToD2M/TTIRToD2M.cpp`
   `toLayoutOperandsAndResults` applied `memorySpaces[role]` (the
   global `default-input/output-memspace`) uniformly to every operand,
   so MOLA could only express placement at input/output-*role*
   granularity. Adds `resolveMolaMemSpace(operand, roleDefault)`: reads
   a per-tensor `mola.memspace` = "l1"|"dram" override from the func
   argument (leaf inputs) or defining op (intermediates) and uses it in
   place of the role default; no attr ⇒ unchanged. This is the only
   correctness-preserving placement seam — it runs *before* the
   `d2m.generic` data-movement region (remote_load/mcast vs local read)
   is generated, so movement codegen is derived consistently from the
   chosen memory space. (A post-ttir-to-d2m `MetalLayoutAttr` flip
   desyncs the already-baked addressing and is silently wrong — see
   `lib/Conversion/MolaAnnotatePlacement.cpp`.) MOLA sets the attr from a
   TTIR-level step in `TTBackend` driven by `mola.target`. Hardware-
   verified: `matmul_320x320` correct (output 320) across all four
   per-tensor placements with four distinct flatbuffers. Required for
   genuine per-tensor MOLA-controlled placement.

10. **`13-ttmetal-flatbuffer-dense-resource-2026-06-03.patch`** — `lib/Target/TTMetal/TTMetalToFlatbuffer.cpp`
    `memrefGlobalOpToFlatbufferByteVector` unconditionally did
    `cast<DenseElementsAttr>(globalOp.getInitialValueAttr())`, which
    asserts when a real (non-splat) weight enters as `ttir.constant`
    with a `dense_resource<...>` value: MOLA lowers it to
    `arith.constant` → `memref.global` whose initial value is a
    `DenseResourceElementsAttr`. Adds a branch that emits the resource's
    raw blob bytes directly (already element-order/endianness correct),
    mirroring the TTNN translator's `ConstantOp` handling. Required for
    QKV / real-weight Llama on the TTMetal chain. Pairs with patch 12's
    `D2MConstantOpRewriter` (the front-end half).

Note: patch 12 also now carries the `D2MConstantOpRewriter` (non-splat
constant → arith.constant) and the `D2MTensorManipulationOpRewriter`
reshape view-memspace fix (a `d2m.view_layout` must preserve memory
space; result layout now takes the input operand's memspace, matching
`D2MPermuteRewriter`) — both required for QKV's transpose+reshape weight
path. See `cee5c93` on the `mola-local` branch.

11. **`20-toplevel-no-reuse-decode-default-2026-06-08.patch`** —
    `lib/Dialect/D2M/Transforms/Allocate.cpp`. Top-level no-reuse decode
    default — the live remaining member of the CB live-range fix series.
    (Its predecessors `17` and `19` are SUPERSEDED by patch 21 and have been
    moved to `superseded/`; they are no longer auto-applied.)

12. **`21-cb-grid-occupancy-gate-2026-06-08.patch`** — `lib/Dialect/D2M/
    Transforms/Allocate.cpp`. The #129 fix: gate disjoint CB allocation on
    combiner GRID OCCUPANCY (cores spanned), not program size. Low occupancy
    ⇒ data-independent siblings run concurrently and need disjoint per-core
    CB scratch (else stale-L1 clobber on re-dispatch); high occupancy ⇒
    siblings serialize and the protective window suffices. Auto-by-capacity,
    budget-bounded. Makes all 9 components + the dim=2048 TinyLlama block
    decode bf16-exact on EVERY dispatch by default.

13. **`22-activation-reuse-fusion-l1-pin-2026-06-08.patch`** — `lib/
    Conversion/TTIRToD2M/TTIRToD2M.cpp`. The two MOLA memory-orchestration
    optimizations (default-on, opt out with `MOLA_TT_L1_PIN_REUSE=0` /
    `MOLA_TT_FUSE_DRAM=0`). `molaPinActivations` ranks matmul-LHS activations
    (opt #1, reuse) + elementwise-op input intermediates (opt #2, fusion) by
    DRAM bytes saved and pins the top into L1 within a reserved budget; the
    matmul/elementwise rewriters propagate `mola.memspace=l1` onto the adapted
    operand at layout time (remapping-proof). Cuts DRAM round-trips 18–64%
    across QKV/attention/SwiGLU/block, decode-exact. See
    `docs/archive/mem-opts-benchmark-2026-06-08.md`.

## Live patches added post-bump (previously undocumented)

- **`11-f3-shared-mlir-stablehlo-funcnest-2026-05-12.patch`** — build/link
  correctness. The "F3" fix: `libMLIR.so` link ordering (`-Wl,--no-as-needed`
  so MLIR/LLVM symbols resolve via the shared lib before static archives pull
  duplicate `cl::opt` initializers) plus a STABLEHLO `func.func` nesting fix and
  a `TokenType` include. Unblocks the `mola-c` link on the bumped toolchain.

- **`14-grid-unit-batch-block-sharded-2026-06-03.patch`** — grid selection.
  Lets unit-batch block-sharded tensors pick a valid grid; the effective-shape
  drives only the rank decision while the real grid still uses the full physical
  shape. Required for Llama 4D activations on the TTMetal chain.

- **`15-l1-sharded-invariant-2026-06-07.patch`** — layout correctness. Enforces
  the invariant that L1-tagged tensors must be `Sharded` (an L1+Interleaved
  combination degenerated a consumer to an input passthrough = silent wrong
  output). Correctness-load-bearing for any L1-placed operand.

- **`16-layout-xform-rank-reconcile-2026-06-07.patch`** — layout transform.
  Reconciles affine-map rank when a layout transform changes tensor rank, so the
  emitted transform maps stay well-formed. Required for the reshape/transpose
  weight paths (QKV).

## Reverted patches (do not re-introduce)

The following patches were attempted and **reverted** because they
addressed only compile-time symptoms of a deeper runtime issue:

- **patch 07** (TTMetalToFlatbuffer alias d2m.stream_layout) — fixed
  the compile-time `Encountered unsupported op` crash for
  `mola-c --enable-capacity-demote`, but the runtime then crashed at
  `executor_utils.h:649` (`meshBuffer.size() == tensorDesc.sizeBytes()`)
  due to host↔device buffer-size mismatch when the demoted IR's
  surviving stream_layout results feed into enqueue_write/read.
- **patch 08** (GridSelection stream-of-stream chain prevention) —
  walked back through chained stream_layouts in
  `insertTTNNDRAMStreams`, eliminating direct chains pre-bufferize,
  but the runtime crash persisted because the size-mismatch is a
  separate bufferDesc/tt-metal-runtime concern that's unaffected by
  IR-level chain prevention.

The actual fix for `--enable-capacity-demote` runtime requires
upstream tt-mlir engagement on tt-metal's buffer-size checks for
demote-introduced streams. This is out of scope for the current
session — `--enable-capacity-demote` remains compile-time-only
without these reverted patches (lit-tested at
`test/Conversion/mola-capacity-demote.mlir`).

### 33 — unaligned innermost reshape stages its operand through L1 (2026-09-05)

`33-d2m-unaligned-reshape-l1-staging-2026-09-05.patch` —
`lib/Conversion/TTIRToD2M/TTIRToD2M.cpp`,
`test/ttmlir/Conversion/TTIRToD2M/reshape_unaligned_inner_l1_staging.mlir`

**The phi2 defect** (`docs/plans/d2m-slice-tile-alignment-2026-08-25.md`, upstream
issue #53). A `ttir.reshape` that re-partitions the innermost dimension into or
out of an extent that is not a multiple of the tile width is miscompiled when its
operand lives in DRAM: the view's addresses are correct in every inspectable
layer, yet the device reads each row start truncated to a 32-element granule.
Six hypotheses were eliminated by measurement; the one positive localisation
is that the identical view over an **L1** operand is exact (extents 16, 40, 80
and the inverse merge).

The patch does exactly that: `D2MTensorManipulationOpRewriter` lays such a
reshape's operand out in L1 by its own `to_layout` (a plain copy under an
identity view, not subject to the defect) and takes the view over the L1
buffer. Aligned reshapes and reshapes that keep the innermost extent use the
role default as before. Extent 20 stays wrong in L1 (0.50) — a second fault in
the same family that no corpus model exercises.

Measured on Blackhole (cosine against torch; TTNN on the identical source):

| reshape | before | after | ttnn |
|---|---|---|---|
| `[1,128,640] -> [1,128,16,40]` | 0.26028 | **1.00000** | 1.00000 |
| `[1,128,16,40] -> [1,128,640]` | 0.27348 | **1.00000** | 1.00000 |
| `[1,128,2560] -> [1,128,32,80]` (phi2's) | 0.49870 | **1.00000** | 1.00000 |
| `[1,128,640] -> [1,128,20,32]` (aligned control) | 1.00000 | 1.00000 | 1.00000 |
| `[1,128,640] -> [1,128,32,20]` (extent 20, known gap) | 0.12887 | 0.50387 | 1.00000 |
| phi2 block 0, corpus export, auto-grid | 0.90094 | **0.99967** | 0.99994 |

Blast radius on the corpus is exactly phi2 (the 26 cached exports were scanned
for the predicate). Cost: one extra DRAM->L1 copy per such reshape; phi2 fits.
The fork's lit test could not be executed on the MOLA box (`ttmlir-opt` does not
link there); MOLA carries two runnable integration tests over `mola-c`
(`test/Integration/d2m-unaligned-reshape-l1-staging.mlir`,
`test/Integration/d2m-aligned-reshape-keeps-dram.mlir`) that pin the operand's
`to_layout` memory space.

### 32 — scalar `pow` exponent is float bits, not an integer (2026-08-27)

`32-d2m-pow-exponent-float-bits-2026-08-27.patch` —
`lib/Conversion/D2MToTTKernel/D2MToTTKernel.cpp`

**D2M computed `x ** 4.2e-45` for every `x ** k`.** tt-metal's `power_tile`
documents `param0` as *"the exponent as IEEE 754 float bits"*, and
`calculate_unary_power` (`ckernel_sfpu_unary_power.h`) opens with
`Converter::as_float(exponent)`. The scalar-pow lowering emitted
`arith.fptosi`, so `3.0` reached the kernel as the integer `3` and
`as_float(3)` is a denormal ~4.2e-45 — the SFPU raised every element to
approximately the zeroth power and returned ~1. No crash, no diagnostic.

The fix is one call: use `scalarToI32Bits`, the helper every OTHER scalar binop
in the same rewriter already uses. The comment being replaced —
`// For power, convert float value to integer (not bitcast)` — was the bug
stated as if it were a reason.

Measured on Blackhole (cosine against torch; vendor TTNN on the identical
StableHLO is the live control, and `x*x*x` the no-`power` control):

| case | before | after | ttnn |
|---|---|---|---|
| `pow(x,2)` | 1.00000 | 1.00000 | 1.00000 |
| `pow(x,3)` | **-0.00191** | **0.99999** | 1.00000 |
| `pow(x,4)` | **0.29251** | **0.99999** | 0.99999 |
| `pow(abs(x)+1, 0.5)` | truncated to `x**0` | 0.99999 | 1.00000 |
| `pow(abs(x)+1, -1.0)` | NaN bit pattern | 0.99999 | 1.00000 |
| `x*x*x` | 0.99999 | 0.99999 | 1.00000 |
| HF `NewGELUActivation` | 0.99971 | 0.99999 | 1.00000 |

(The 0.99971 for `gelu_new` is the figure recorded in
`docs/plans/d2m-power-defect-2026-08-25.md`; the rest of the "before" column was
re-measured in this session on the same build before the patch was applied.)

**Why it survived:** tt-mlir's `scalar_binary_ops.mlir` asserted only that
`power_tile` was CALLED, never how its argument was encoded, so it passed the
whole time. The patch strengthens that test to pin the encoding and adds a
fractional-exponent case. MOLA carries its own end-to-end gate at
`test/Integration/d2m-pow-exponent-encoding.mlir`, which greps the kernel source
embedded in the emitted flatbuffer — verified to FAIL when the expected constant
is perturbed.

### 31 — OpModel reads resource-backed constants (2026-08-10)

`31-opmodel-resource-constants-2026-08-10.patch`

**The vendor optimizer could not run on any real checkpoint.** tt-mlir's own
memory-layout optimizer (`optimization-level=1|2`) needs `TTMLIR_ENABLE_OPMODEL`,
and with it enabled the first HuggingFace model aborts in
`TTNNOpModel.cpp::getElementType`:

```
Assertion `false && "Unknown constant value attribute type"' failed.
```

Three functions there enumerate `DenseElementsAttr` and `SplatElementsAttr` and
reject everything else. Weights imported from PyTorch are neither: the importer
stores bulk data as **`DenseResourceElementsAttr`**, out-of-line, rather than
inlining megabytes of hex into the IR. So the optimizer worked on hand-written
test IR and on nothing a user would actually compile.

| function | change |
|---|---|
| `getElementType` | ask the attribute's `ShapedType` instead of enumerating subclasses — every `ElementsAttr` has one, so the two special cases were already redundant |
| `getRawDataFromElementsAttr<T>` | read the resource blob and `memcpy` it out |
| `getRawDataFromElementsAttr<bfloat16>` | same, element-wise, since bf16 arrives as raw `uint16_t` bits |

Both data paths **size-check the blob against the element count** before reading
it. The blob is raw host bytes with no length guarantee attached at the use site,
so an unchecked read would silently pick up whatever follows it in memory — a
wrong-numbers bug of exactly the kind this project keeps finding. A mismatch
returns an `llvm::Error`, which the caller already handles as a declined
optimization. The bf16 path uses `memcpy` per element rather than a
`reinterpret_cast` read because the blob carries no alignment guarantee for
`uint16_t`.

Needs `mlir/IR/DialectResourceBlobManager.h`: `BuiltinAttributes.h` only
forward-declares `DialectResourceBlobHandle`.

Measured on SmolLM-135M, sequence 128, Blackhole: before, abort; after,
`optimization-level=1` compiles in 3.0 s with **0** validation failures (it had
been 7 `ttnn.constant` failures, each fatal to the pipeline).

This is what makes MOLA's placement comparable against the vendor's own
optimizer rather than against a pipeline with every optimizer pass switched off.

### 30 — reblock precondition recoverable, not fatal (2026-08-04)

`30-reblock-precondition-recoverable-2026-08-04.patch`

**Root cause of the D2M sequence-length ceiling.** `reblockShapedType` computes
each new shard as `(oldGrid[i] * oldShard[i]) / gridDim[i]` and asserts the
division is exact — a `ShapedType` carries one shard shape for every core and
cannot express an uneven split. The precondition is genuine, but it is enforced
with `TT_assert`, which **aborts the process** rather than reporting something a
caller can act on.

Upstream treats it as unreachable because grid selection searches for the largest
*divisor* of the tile count. It is reachable: the divisor is chosen against one
shape, and by the time reblocking runs the shape has been through interval
collapse and alignment, so it need not still divide. SmolLM-135M aborts this way
at sequence 320 and above **with no user grid override at all**.

Adds `utils::canReblockShapedType` — the same conditions as the asserts, returned
as a `bool` — and asks before reblocking at the sites that *choose* a grid:

| site | status |
|---|---|
| `D2MOps.cpp`, `GenericOp::withParallelization` | the site actually observed to abort, confirmed by stack trace. Its lambda already returns `FailureOr` and already reports an underivable grid shape, so an unreblockable operand is reported the same way and every caller already handles it. |
| `GridSelection.cpp`, two call sites | same precondition, same conservative decline. **Not** observed to fire; defensive. |

Declining is the conservative direction: the op keeps the grid it already has,
which is trivially reblockable. A less parallel grid costs performance; an abort
costs the compile.

Measured on SmolLM-135M, stock automatic path, no MOLA flags:

    before   seq 320, 512   rc=134, "Assertion ... failed, was `2 == 0`"
    after    seq 320, 512   rc=1,   "cannot reblock operand N ... onto grid [...]"

With `MOLA_TT_AUTO_GRID=1` the search recovers and all three compile — seq 128
and 320 on grid 4,12 and seq 512 on 8,13, at cosine 0.99970 / 0.99969 / 0.99957
against a PyTorch reference with a shuffled-reference control firing in each
case. Every artifact was dispatched, not merely compiled.

**Rebuild note.** `scripts/build.sh` does *not* rebuild the vendored tt-mlir, so
a change here needs `ninja TTMLIRCompiler` in `build/ttmlir`. Skipping it makes
the guards appear to do nothing — two test rounds were run against a stale
library before this was noticed.

**Durability (updated 2026-08-04).** Now a commit in the submodule, on branch
`mola-local-patched-2026-08-04` (`ff44e630`), together with a preceding commit
`b977b3fb` that persists the previously applied-but-uncommitted records 11–29.
Both branch from the pinned commit `48bb0180` rather than from the `mola-local`
tip, which is four commits ahead — moving to the tip would change the compiler
that the grid-selection and feasibility numbers were measured against, and that
is a separate decision.

**These two commits are NOT yet on the fork remote.** The parent repo's gitlink
has been bumped locally but must not be pushed before
`git -C third_party/tt-mlir push origin mola-local-patched-2026-08-04`, or a
recursive clone will reference a commit that does not exist.

## Fork + patches (belt-and-suspenders)

As of 2026-06-08 the tt-mlir submodule is pinned to a personal fork
(`amirnd51/tt-mlir`, branch `mola-local`) which carries all of these
changes as commits — so a recursive clone gets the full toolchain
directly. These patch files are kept in sync as a **redundant, reviewable
record**: they apply cleanly on top of the upstream base via
`scripts/apply-tt-mlir-patches.sh` (idempotent — skips already-applied),
so the changes are reproducible and auditable even without the fork. Once
stable, individual patches should be contributed upstream.

## deferred-on-bump/ — AUDITED AND CLOSED (2026-08-24)

The six patches deferred at the 2026-05-11 bump (01–04, 06, 08) are **all
resolved**. None needs re-authoring. That directory's own README still says "the
`--backend tt` chain currently regresses without these" — that sentence is STALE,
written before the fork's June commits, and is contradicted by the corpus run
below.

Checked against the tree at tt-mlir `a26f9e5899`, not assumed:

| # | Status | Evidence |
|---|---|---|
| `01-ttir-to-d2m-branch-on-memspace` | **upstream now does it** | `TTIRToD2M.cpp` `extractLayoutInfo` branches `BufferType::DRAM ? DeviceDRAM : DeviceL1` and checks `TensorMemoryLayout::Interleaved` |
| `02-grid-selection-dram-interleaved-bridge` | **target refactored out; symptom gone** | `insertTTNNDRAMStreams` / `hasTTNNOperands` have 0 refs in `GridSelection.cpp`. The memory-layout tagging it existed to fix is handled by the fork's 2026-06-07 patch at `GridSelection.cpp:331` ("Interleaved is only correct for..."), and gemma2-2b — whose registry blocker cited the DRAM-Interleaved→L1 bridge — now compiles (169.2 MB) and dispatches |
| `03-lower-to-layout-staging-buffer-sharded` | **re-authored as a fork commit** | `LowerToLayout/Plan.cpp:248` "buffer must be memory_layout=Sharded — Interleaved is only legal for..." (commit `f3918b722f`) |
| `04-fullop-bufferize-element-type-cast` | **obsolete** | `d2m::FullOp` no longer exists — 0 refs in `D2MOps.td` |
| `06-flatbuffer-bf16-fp16-serialization` | **upstream now does it** | `TTMetalToFlatbuffer.cpp:1091` bitcasts 16-bit floats to `uint16_t`, which is what the patch added |
| `08-ttir-constant-to-d2m` | **re-authored as a fork commit** | `TTIRToD2M.cpp:3450` `D2MConstantOpRewriter` — "non-splat case (real model weights via dense_resource)" (commit `3a3f24b695`) |

Corroborated end to end rather than by reading: all six corpus checkpoints
compile AND dispatch on Blackhole via `--backend tt`, and the two Llama-family
models land inside their bf16 conditioning floor (smollm-135m 0.99994,
tinyllama-1.1b 0.99997). The residual accuracy gap on non-Llama attention is in
STOCK D2M lowering, not in these patches: `d2m-on` vs `--no-placement` gives
byte-different artifacts and identical output to five decimals.

The `.patch` files stay on disk as the historical record. Nothing in
`deferred-on-bump/` is an open action any more.
