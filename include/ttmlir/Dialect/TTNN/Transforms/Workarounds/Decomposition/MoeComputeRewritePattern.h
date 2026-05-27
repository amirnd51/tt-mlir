// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_MOECOMPUTEREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_MOECOMPUTEREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Fixes up MoeComputeOp result encodings that the generic operand-workarounds
// framework cannot express: the three height-sharded outputs
// (per_expert_total_tokens, tilize_output, matmul_output) need a
// {numWorkerCores, 1} virtual grid in their TTNNLayoutAttr so that the
// runtime's insertTTNNTensorAndValidate accepts the kernel's allocated
// memory config (HEIGHT_SHARDED across all worker cores).
//
// Weight inputs (matmul_w0_w1_tensor, matmul_w2_tensor) arrive
// pre-packed from prepare_moe_compute_weights and are intentionally not
// touched here — see TTNNPrepareMoEComputeWeights pass and the runtime helper
// chain.
class MoeComputeRewritePattern : public OpRewritePattern<ttnn::MoeComputeOp> {
public:
  using OpRewritePattern<ttnn::MoeComputeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::MoeComputeOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_MOECOMPUTEREWRITEPATTERN_H
