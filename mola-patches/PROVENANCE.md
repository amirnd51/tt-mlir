# Why these patch files live here

These are the MOLA-local patches to tt-mlir. Their **content** is already in this
fork as commits on `mola-local-patched-2026-08-24` — that branch, which MOLA's
submodule gitlink points at, is the authoritative patched toolchain, and a
recursive clone gets the patched compiler without needing anything in this
directory.

The `.patch` files are the *reviewable record*: what changed, and why. They used
to live only in the MOLA working copy, where `.gitignore` excluded
`third_party/tt-mlir-patches/*.patch` from version control entirely. That made
them a single-machine artifact — losing the box lost the record, and a fresh
clone of MOLA ran `scripts/apply-tt-mlir-patches.sh` to be told
"no .patch files found, nothing to do".

Copied here so the record travels with the fork it describes.

`README.md` beside this file is the audit narrative (what each patch changes and
why), copied from `third_party/tt-mlir-patches/README.md` in MOLA, which stays
the tracked upstream copy.
