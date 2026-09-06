# Superseded / obsolete tt-mlir patches (NOT auto-applied)

These patch files are kept as a historical record but are **deliberately
excluded** from `scripts/apply-tt-mlir-patches.sh` (which globs only the
top-level `*.patch`). They no longer apply against the pinned submodule and
would make the apply script `exit 1` on a fresh checkout.

- **`05-rank-normalization-dense-resource.patch`** — OBSOLETE. Targeted
  `lib/Dialect/TTIR/Transforms/RankNormalization.cpp`, which upstream **deleted**
  (commit `daf610169b`, "[d2m] remove rank norm"). The `updateConstantValueAttr`
  function it patched no longer exists; the dense-resource rank-sync concern it
  addressed is moot on the bumped tree.

- **`17-cb-window-live-ranges-2026-06-07.patch`**,
  **`19-cb-combiner-disjoint-optin-2026-06-08.patch`** — SUPERSEDED. The CB
  allocation fix evolved 17 → 19 → **21** (`21-cb-grid-occupancy-gate`) as
  sequential commits on the `mola-local` fork branch. Patch 21 (still in the
  auto-applied top-level dir) carries the final form; 17/19 are intermediate
  snapshots whose exact diffs are no longer present in the submodule, so they
  fail both the forward- and reverse-apply checks.

The submodule is pinned to the `amirnd51/tt-mlir` fork (`mola-local`), which
already carries the live changes as commits, so the authoritative toolchain comes
from the fork — these records are belt-and-suspenders only.
