// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/SortOpRewritePattern.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// The decomposition is O(n^2) in the sort-dim length; beyond this we leave the
// (faster, but index-buggy) ttnn.sort in place rather than blow up the graph.
static constexpr int64_t kMaxArgsortDecompDim = 4096;

// Every intermediate here changes shape and/or dtype, so each needs a fresh
// TTNN layout encoding derived from `base` (shape first, then dtype).
static RankedTensorType typedShape(RankedTensorType base,
                                   ArrayRef<int64_t> shape,
                                   ttcore::DataType dt) {
  RankedTensorType withShape =
      ttnn::utils::RankedTensorTypeFactory::create(base, shape);
  return ttnn::utils::RankedTensorTypeFactory::create(withShape, dt);
}

LogicalResult
SortOpRewritePattern::matchAndRewrite(ttnn::SortOp op,
                                      PatternRewriter &rewriter) const {
  RankedTensorType inputType = op.getInput().getType();

  // Float keys sort correctly; only integer keys hit the bf16-compare bug, and
  // only the indices output is wrong, so there is nothing to fix otherwise.
  if (!mlir::isa<IntegerType>(inputType.getElementType())) {
    return failure();
  }
  if (op.getIndices().use_empty()) {
    return failure();
  }

  int64_t rank = inputType.getRank();
  int64_t dim = op.getDim();
  if (dim < 0) {
    dim += rank;
  }
  // Last-dim only keeps the reshape-to-2D below trivial; other dims are rare
  // for integer argsort and fall back to the kernel.
  if (dim != rank - 1) {
    return failure();
  }

  ArrayRef<int64_t> inShape = inputType.getShape();
  int64_t n = inShape[dim];
  if (n <= 0 || n > kMaxArgsortDecompDim) {
    return failure();
  }
  int64_t R = inputType.getNumElements() / n;

  Location loc = op.getLoc();
  MLIRContext *ctx = rewriter.getContext();
  bool descending = op.getDescending();
  const auto f32 = ttcore::DataType::Float32;
  int32_t Ri = static_cast<int32_t>(R), ni = static_cast<int32_t>(n);
  auto i32arr = [&](std::initializer_list<int32_t> v) {
    return rewriter.getI32ArrayAttr(SmallVector<int32_t>(v));
  };

  RankedTensorType idxType = op.getIndices().getType();
  ttcore::DataType inDt =
      ttcore::elementTypeToDataType(inputType.getElementType());

  // f32 is the whole point: it represents the integer keys exactly (<= 2^24),
  // so the comparisons below never collide the way ttnn.sort's bf16 compare
  // does. Flatten the batch dims so the sort axis is a simple 2D row.
  Value in2D =
      rewriter.create<ttnn::ReshapeOp>(loc, typedShape(inputType, {R, n}, inDt),
                                       op.getInput(), i32arr({Ri, ni}));
  Value xf = rewriter.create<ttnn::TypecastOp>(
      loc, typedShape(inputType, {R, n}, f32), in2D,
      ttcore::DataTypeAttr::get(ctx, f32));

  // Pairwise key matrix [R,n,n]: xiB[r,i,j]=x[i], xjB[r,i,j]=x[j].
  RankedTensorType t3f = typedShape(inputType, {R, n, n}, f32);
  Value xi = rewriter.create<ttnn::ReshapeOp>(
      loc, typedShape(inputType, {R, n, 1}, f32), xf, i32arr({Ri, ni, 1}));
  Value xj = rewriter.create<ttnn::ReshapeOp>(
      loc, typedShape(inputType, {R, 1, n}, f32), xf, i32arr({Ri, 1, ni}));
  Value xiB = rewriter.create<ttnn::RepeatOp>(
      loc, t3f, xi, ttnn::ShapeAttr::get(ctx, {1, 1, n}));
  Value xjB = rewriter.create<ttnn::RepeatOp>(
      loc, t3f, xj, ttnn::ShapeAttr::get(ctx, {1, n, 1}));

  // Position matrices iIota[r,i,j]=i and jIota[r,i,j]=j, used for the stable
  // tiebreak and the inverse-permutation step.
  GetDeviceOp device = ttnn::utils::getOrInsertDevice(rewriter, op);
  ttcore::DataTypeAttr f32Attr = ttcore::DataTypeAttr::get(ctx, f32);
  RankedTensorType arType = typedShape(inputType, {n}, f32);
  ttnn::LayoutAttr arLayout = ttnn::LayoutAttr::get(
      ctx, mlir::cast<ttnn::TTNNLayoutAttr>(arType.getEncoding()).getLayout());
  Value ar = rewriter.create<ttnn::ArangeOp>(loc, arType, device,
                                             /*start=*/0, /*end=*/n, /*step=*/1,
                                             f32Attr, arLayout);
  Value arRow = rewriter.create<ttnn::ReshapeOp>(
      loc, typedShape(inputType, {1, 1, n}, f32), ar, i32arr({1, 1, ni}));
  Value arCol = rewriter.create<ttnn::ReshapeOp>(
      loc, typedShape(inputType, {1, n, 1}, f32), ar, i32arr({1, ni, 1}));
  Value jIota = rewriter.create<ttnn::RepeatOp>(
      loc, t3f, arRow, ttnn::ShapeAttr::get(ctx, {R, n, 1}));
  Value iIota = rewriter.create<ttnn::RepeatOp>(
      loc, t3f, arCol, ttnn::ShapeAttr::get(ctx, {R, 1, n}));

  Value cmp =
      descending
          ? rewriter.create<ttnn::GreaterThanOp>(loc, t3f, xjB, xiB).getResult()
          : rewriter.create<ttnn::LessThanOp>(loc, t3f, xjB, xiB).getResult();

  // Break ties by original index (j < i) so equal keys sort stably; OR it into
  // the strict compare via max (operands are 0/1).
  Value eq = rewriter.create<ttnn::EqualOp>(loc, t3f, xjB, xiB);
  Value jLtI = rewriter.create<ttnn::LessThanOp>(loc, t3f, jIota, iIota);
  Value tie = rewriter.create<ttnn::MultiplyOp>(loc, t3f, eq, jLtI);
  Value lessOrTie = rewriter.create<ttnn::MaximumOp>(loc, t3f, cmp, tie);

  // rank[r,i] = #keys ordered before i = its sorted position.
  RankedTensorType t2f = typedShape(inputType, {R, n}, f32);
  Value rank2D = rewriter.create<ttnn::SumOp>(loc, t2f, lessOrTie,
                                              /*keep_dim=*/false, i32arr({2}));

  // Invert the rank permutation: argsort[r,k] = sum_i i * (rank[r,i] == k).
  Value rankCol = rewriter.create<ttnn::ReshapeOp>(
      loc, typedShape(inputType, {R, n, 1}, f32), rank2D, i32arr({Ri, ni, 1}));
  Value rankB = rewriter.create<ttnn::RepeatOp>(
      loc, t3f, rankCol, ttnn::ShapeAttr::get(ctx, {1, 1, n}));
  Value onehot = rewriter.create<ttnn::EqualOp>(loc, t3f, rankB, jIota);
  Value prod = rewriter.create<ttnn::MultiplyOp>(loc, t3f, onehot, iIota);
  Value argF2D = rewriter.create<ttnn::SumOp>(loc, t2f, prod,
                                              /*keep_dim=*/false, i32arr({1}));

  ttcore::DataType idxDt =
      ttcore::elementTypeToDataType(idxType.getElementType());
  Value argIdx2D = rewriter.create<ttnn::TypecastOp>(
      loc, typedShape(inputType, {R, n}, idxDt), argF2D,
      ttcore::DataTypeAttr::get(ctx, idxDt));
  SmallVector<int32_t> outShapeI32(inShape.begin(), inShape.end());
  Value indicesND = rewriter.create<ttnn::ReshapeOp>(
      loc, idxType, argIdx2D, rewriter.getI32ArrayAttr(outShapeI32));

  // Values are correct from ttnn.sort, so keep it for them. Its own indices
  // are unused, so use_empty() above stops this pattern from re-matching it.
  auto valuesSort = rewriter.create<ttnn::SortOp>(
      loc, TypeRange{op.getValues().getType(), op.getIndices().getType()},
      op.getInput(), op.getDimAttr(), op.getDescendingAttr(),
      op.getStableAttr());

  rewriter.replaceOp(op, {valuesSort.getValues(), indicesND});
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
