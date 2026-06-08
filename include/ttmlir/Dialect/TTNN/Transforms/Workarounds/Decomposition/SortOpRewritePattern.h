// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SORTOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SORTOPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Works around a tt-metal bug where `ttnn.sort` compares keys in bf16 (it
// reuses the TopK SFPU sort LLK) and ignores the `stable` flag, so integer keys
// that lack an exact bf16 representation (> 256) collide into ties and the
// returned *indices* become arbitrary -- the sorted values are fine but the
// index permutation is wrong. Models that use argsort indices to (un)permute a
// tensor (e.g. Qwen2.5-VL's vision window-index reverse permute) then get
// scrambled results.
//
// For an integer-keyed sort whose indices result is actually used and whose
// sort-dim is within an O(n^2) budget, this rewrites the indices via
// rank-by-comparison done entirely in f32 (exact for integer magnitudes
// <= 2^24, so it never routes index values through bf16):
//
//   rank[i]    = #{ j : x[j] < x[i] }  +  #{ j : x[j]==x[i] and j<i }  (stable)
//   argsort[k] = sum_i i * (rank[i] == k) (inverse)
//
// The sorted *values* keep coming from `ttnn.sort` (its values are correct).
// Remove once the metal-side fix lands.
// Issue: https://github.com/tenstorrent/tt-metal/issues/46331
class SortOpRewritePattern : public OpRewritePattern<ttnn::SortOp> {
public:
  using OpRewritePattern<ttnn::SortOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::SortOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SORTOPREWRITEPATTERN_H
