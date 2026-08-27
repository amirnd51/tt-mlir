# tt-mlir patches deferred during 2026-05-11 bump

When tt-mlir was bumped from `4490eed83e` (Feb 25) to `9bc476e771`
(May 11, pinning tt-metal `08ce2dc15` + UMD `7b37f8aa15` — fixes the
FW 19.4.2 BH card init issue), the following 6 patches stopped
applying cleanly:

| # | Reason |
|---|---|
| `01-ttir-to-d2m-branch-on-memspace.patch` | TTIRToD2M.cpp line numbers shifted; concept still relevant. |
| `02-grid-selection-dram-interleaved-bridge.patch` | `hasTTNNOperands` + `insertTTNNDRAMStreams` were refactored out of GridSelection.cpp (file 1022 → 775 lines). |
| `03-lower-to-layout-staging-buffer-sharded.patch` | `LowerToLayout.cpp` moved to `LowerToLayout/LowerToLayout.cpp`. |
| `04-fullop-bufferize-element-type-cast.patch` | `d2m::FullOp::bufferize` removed entirely — `d2m::FullOp` no longer exists upstream. Likely obsolete. |
| `06-flatbuffer-bf16-fp16-serialization.patch` | Target function `memrefGlobalOpToFlatbufferByteVector` shifted (847 → 947). Re-author with new offsets. |
| `08-ttir-constant-to-d2m.patch` | Targeted line 2354 of TTIRToD2M.cpp (file grew to 4328 lines). Need to re-author as a sibling pattern next to `D2MConstantFillOpRewriter` at line 2826. |

All six are TTMetal-chain only (`--backend tt`) — none gate the
`--backend tt-ttnn` runtime path that's used for Llama on Blackhole.
Active patches (`05`, `10`) cover the TTNN flow.

The `--backend tt` chain currently regresses without these. Re-author
each when the TTMetal chain runtime work resumes.

The original Feb-25-era patch contents are preserved here verbatim
for reference (they cannot be `git apply`d against the bumped source).
