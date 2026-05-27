// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/MoeComputeRewritePattern.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <numeric>

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult
MoeComputeRewritePattern::matchAndRewrite(MoeComputeOp srcOp,
                                          PatternRewriter &rewriter) const {
  // tt-metal's `MoEComputeDeviceOperation::compute_output_specs` derives the
  // canonical output shapes/layouts/dtypes from the device's compute grid and
  // the operands. The frontend's caller can't know these (especially the
  // device-grid-derived shapes), so we recompute them here and rebuild every
  // output's RankedTensorType to match — otherwise the runtime's
  // `checkTensorRefMatchesTTNNTensor` asserts on a layout/storage/dtype
  // mismatch when tt-metal returns tensors that disagree with the flatbuffer.
  //
  // Inputs 1 (tilize_expert_indices) and 2 (tilize_expert_scores) must be L1
  // HEIGHT_SHARDED on the kernel's tilize drain core —
  // moe_compute_program_factory globally-allocates CBs against them and
  // otherwise throws "Only L1 buffers can have an associated circular buffer".
  auto tilizeOutType =
      mlir::cast<RankedTensorType>(srcOp.getTilizeOutput().getType());
  auto tilizeOutEncoding =
      mlir::cast<TTNNLayoutAttr>(tilizeOutType.getEncoding());

  // Idempotency: once tilize_output is L1 HEIGHT_SHARDED with the right grid
  // we are done. The generic factory only sets layout+dtype, so a fresh op
  // will not yet be height-sharded.
  if (tilizeOutEncoding.getBufferType() == BufferType::L1 &&
      tilizeOutEncoding.getMemLayout() &&
      tilizeOutEncoding.getMemLayout().getValue() ==
          TensorMemoryLayout::HeightSharded) {
    return failure();
  }

  // The system descriptor reports the physical worker grid (8x10 on Wormhole
  // 6U Galaxy), but at runtime the dispatch system reserves cores and exposes
  // a smaller rectangular compute_with_storage grid of (x=7, y=10) = 70
  // cores. The moe_compute kernel and its weight-prep ops must place their
  // height-sharded outputs entirely within that rectangle: e.g. an L-shaped
  // 8x8 + 6x1 placement (canonical for 70 shards on an 8x9 bbox) reaches core
  // (7,8), which is outside the (7,10) compute grid and triggers a TT_FATAL
  // when `interleaved_to_sharded` builds a circular buffer there.
  //
  // Use the (7,10) rectangle explicitly. TODO: expose this grid via the
  // system descriptor so we don't hardcode the Galaxy dispatch layout.
  MLIRContext *ctx = srcOp.getContext();
  constexpr int64_t kComputeGridX = 7;
  constexpr int64_t kComputeGridY = 10;
  int64_t numWorkerCores = kComputeGridX * kComputeGridY;
  CoreRangeSetAttr computeGridRangeSet = CoreRangeSetAttr::get(
      ctx, CoreRangeAttr::get(ctx, CoreCoordAttr::get(ctx, 0, 0),
                              CoreCoordAttr::get(ctx, kComputeGridX - 1,
                                                 kComputeGridY - 1)));

  // tt-metal's moe_compute kernel attaches globally-allocated CBs for the
  // expert-indices and expert-scores tensors onto each of the 4 tilize
  // cores, all pointing at a single backing buffer. That buffer must live
  // on the drain core — the first entry of `max_tilize_cores` in
  // moe_compute_program_factory.cpp, i.e. (6, 9) on WH. The reference 6U
  // test does exactly this (test_moe_compute_6U.py:1294 + 1321):
  //   tilize_drain_core = CoreRangeSet({CoreRange(CoreCoord(6,9), (6,9))})
  //   expert_indices_mem_config = create_sharded_memory_config(
  //       tilize_drain_core, [total_tokens, selected_experts_k], uint16)
  // Sharding across 4 cores instead sizes the per-core CB at the full page
  // count while only allocating shard-sized L1 buffers per core, which
  // trips circular_buffer_config.cpp:164 ("total_size > max_size_").
  CoreRangeSetAttr tilizeDrainCoreRangeSet = CoreRangeSetAttr::get(
      ctx, CoreRangeAttr::get(ctx, CoreCoordAttr::get(ctx, 6, 9),
                              CoreCoordAttr::get(ctx, 6, 9)));
  constexpr int64_t kNumDrainCores = 1;

  // Constants pulled from
  // ttnn/cpp/.../moe_compute/device/moe_compute_device_operation.cpp:
  //   constexpr auto TOKEN_SIZE = 32;
  //   constexpr auto DOUBLE_BUFFER_SIZE = 2;
  // and the platform-wide L1 alignment (16B on Wormhole).
  constexpr int64_t kL1Alignment = 16;
  constexpr int64_t kDoubleBufferSize = 2;
  constexpr int64_t kTokenSize = 32;
  constexpr int64_t kU32Bytes = sizeof(uint32_t);
  auto alignUp = [](int64_t v, int64_t a) { return ((v + a - 1) / a) * a; };

  // Derive the same quantities `compute_output_specs` reads off the inputs:
  // experts_per_device (from the mapping tensor + cluster mesh size),
  // total_tokens and hidden_size (from the dispatched input tensor).
  auto inputType =
      mlir::cast<RankedTensorType>(srcOp.getTilizeInputTensor().getType());
  auto mappingType =
      mlir::cast<RankedTensorType>(srcOp.getTilizeExpertMappingTensor().getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> mappingShape = mappingType.getShape();

  ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(srcOp.getOperation());
  ArrayRef<int64_t> meshShape = deviceAttr.getMeshShape();
  int64_t numDevices = std::accumulate(meshShape.begin(), meshShape.end(),
                                       int64_t{1}, std::multiplies<int64_t>());

  int64_t experts = mappingShape.back();
  int64_t expertsPerDevice = (experts + numDevices - 1) / numDevices;
  int64_t totalTokens = inputShape[0] * inputShape[1];
  int64_t hiddenSize = inputShape.back();

  // Output 0 (per_expert_total_tokens): UINT32 ROW_MAJOR L1 HEIGHT_SHARDED.
  // Shape `(num_cores, align(epd*sizeof(u32), l1)/sizeof(u32))`.
  int64_t perExpertRowBytes = alignUp(expertsPerDevice * kU32Bytes, kL1Alignment);
  int64_t perExpertRowElems = perExpertRowBytes / kU32Bytes;
  SmallVector<int64_t, 2> output0Shape{numWorkerCores, perExpertRowElems};

  // Output 1 (expert_activation): UINT32 ROW_MAJOR L1 INTERLEAVED.
  // Shape `(1, total_tokens * align((2*epd+1)*sizeof(u32), l1)/sizeof(u32))`.
  int64_t activationRowElems = (2 * expertsPerDevice) + 1;
  int64_t activationRowBytes = alignUp(activationRowElems * kU32Bytes, kL1Alignment);
  int64_t activationTotalBytes = totalTokens * activationRowBytes;
  SmallVector<int64_t, 2> output1Shape{1, activationTotalBytes / kU32Bytes};

  // Output 2 (expert_to_token): UINT32 ROW_MAJOR L1 INTERLEAVED.
  // Shape `(epd, (total_tokens+1) * align(sizeof(u32), l1)/sizeof(u32))`.
  int64_t eTRowBytes = (totalTokens + 1) * alignUp(kU32Bytes, kL1Alignment);
  int64_t eTRowElems = eTRowBytes / kU32Bytes;
  SmallVector<int64_t, 2> output2Shape{expertsPerDevice, eTRowElems};

  // Output 3 (tilize_output): BFLOAT16 TILE L1 HEIGHT_SHARDED.
  // Shape `(num_cores, DOUBLE_BUFFER_SIZE, TOKEN_SIZE, hidden_size)`. With the
  // default collapse intervals, the sharded view collapses to
  // `(num_cores * DOUBLE_BUFFER_SIZE * TOKEN_SIZE, hidden_size)` and divides
  // evenly across `num_cores` height shards as `(64, hidden_size)` per core
  // — matching tt-metal's shard_spec.
  SmallVector<int64_t, 4> output3Shape{numWorkerCores, kDoubleBufferSize,
                                       kTokenSize, hiddenSize};
  // Output 4 (matmul_output): same logical shape as output 3 but ROW_MAJOR
  // (a reinterpret of the same backing buffer).
  SmallVector<int64_t, 4> output4Shape = output3Shape;

  // Build the new encodings. Order matters within TTNNLayoutAttr::Builder:
  // setBufferType / setMemoryLayout / setGridShape each invalidate the
  // CoreRangeSet, so the explicit CRS must be set last.
  auto buildHsEncoding = [&](TTNNLayoutAttr seed, ArrayRef<int64_t> shape,
                             Layout pageLayout) {
    return TTNNLayoutAttr::Builder(seed, shape)
        .setLayout(pageLayout)
        .setBufferType(BufferType::L1)
        .setMemoryLayout(TensorMemoryLayout::HeightSharded)
        .setGridShape({numWorkerCores, 1})
        .setCoreRangeSet(computeGridRangeSet)
        .build();
  };

  auto buildInterleavedEncoding = [&](TTNNLayoutAttr seed,
                                      ArrayRef<int64_t> shape,
                                      Layout pageLayout) {
    return TTNNLayoutAttr::Builder(seed, shape)
        .setLayout(pageLayout)
        .setBufferType(BufferType::L1)
        .setMemoryLayout(TensorMemoryLayout::Interleaved)
        .setGridShape({1, 1})
        .build();
  };

  auto buildTilizeHsEncoding = [&](TTNNLayoutAttr seed,
                                   ArrayRef<int64_t> shape) {
    return TTNNLayoutAttr::Builder(seed, shape)
        .setBufferType(BufferType::L1)
        .setMemoryLayout(TensorMemoryLayout::HeightSharded)
        .setGridShape({kNumDrainCores, 1})
        .setCoreRangeSet(tilizeDrainCoreRangeSet)
        .build();
  };

  auto rebuildResult = [&](Value oldResult, ArrayRef<int64_t> newShape,
                           Layout pageLayout, bool heightSharded) {
    auto t = mlir::cast<RankedTensorType>(oldResult.getType());
    auto seed = mlir::cast<TTNNLayoutAttr>(t.getEncoding());
    auto enc = heightSharded ? buildHsEncoding(seed, newShape, pageLayout)
                             : buildInterleavedEncoding(seed, newShape,
                                                        pageLayout);
    return RankedTensorType::get(newShape, t.getElementType(), enc);
  };

  // Insert a ttnn.to_memory_config converting `oldInput` to an L1 HEIGHT_SHARDED
  // layout on the tilize cores. Result has the same shape/dtype/layout (only
  // memory_config differs).
  auto reshardToTilize = [&](Value oldInput) -> Value {
    auto t = mlir::cast<RankedTensorType>(oldInput.getType());
    auto seed = mlir::cast<TTNNLayoutAttr>(t.getEncoding());
    auto newEncoding = buildTilizeHsEncoding(seed, t.getShape());
    auto newType =
        RankedTensorType::get(t.getShape(), t.getElementType(), newEncoding);
    return rewriter.create<ttnn::ToMemoryConfigOp>(srcOp.getLoc(), newType,
                                                   oldInput);
  };

  SmallVector<Type> newResultTypes(srcOp.getResultTypes());
  newResultTypes[0] = rebuildResult(srcOp.getPerExpertTotalTokens(),
                                    output0Shape, Layout::RowMajor,
                                    /*heightSharded=*/true);
  newResultTypes[1] = rebuildResult(srcOp.getExpertActivation(), output1Shape,
                                    Layout::RowMajor,
                                    /*heightSharded=*/false);
  newResultTypes[2] = rebuildResult(srcOp.getExpertToToken(), output2Shape,
                                    Layout::RowMajor,
                                    /*heightSharded=*/false);
  newResultTypes[3] = rebuildResult(srcOp.getTilizeOutput(), output3Shape,
                                    Layout::Tile,
                                    /*heightSharded=*/true);
  newResultTypes[4] = rebuildResult(srcOp.getMatmulOutput(), output4Shape,
                                    Layout::RowMajor,
                                    /*heightSharded=*/true);

  Value newIndices = reshardToTilize(srcOp.getTilizeExpertIndicesTensor());
  Value newScores = reshardToTilize(srcOp.getTilizeExpertScoresTensor());

  // tt-metal's selective_reduce_combine asserts `num_links * neighbors.size()
  // <= mux_core_range_set.num_cores()`. The reference 6U test
  // (test_moe_compute_6U.py:1574) provides a 3x3 block at (1,1)-(3,3) = 9
  // cores, which fits any num_links/neighbors config the WH path can
  // produce (max num_links=4 * 2 neighbors = 8). Default to that when the
  // caller didn't specify a mux range.
  CoreRangeSetAttr muxCoreRangeSet = srcOp.getMuxCoreRangeSetAttr();
  if (!muxCoreRangeSet) {
    muxCoreRangeSet = CoreRangeSetAttr::get(
        ctx, CoreRangeAttr::get(ctx, CoreCoordAttr::get(ctx, 1, 1),
                                CoreCoordAttr::get(ctx, 3, 3)));
  }

  auto newOp = rewriter.create<MoeComputeOp>(
      srcOp.getLoc(), TypeRange(newResultTypes), srcOp.getTilizeInputTensor(),
      newIndices, newScores, srcOp.getTilizeExpertMappingTensor(),
      srcOp.getMatmulW0W1Tensor(), srcOp.getMatmulW2Tensor(),
      srcOp.getOptionalOutputTensor(), srcOp.getCrossDeviceSemaphore(),
      srcOp.getDevice(), srcOp.getLayerIdAttr(),
      srcOp.getOutputHeightShardDimAttr(), srcOp.getIntermediateSizeAttr(),
      srcOp.getHasBiasAttr(), srcOp.getClusterAxisAttr(),
      srcOp.getActivationFunctionAttr(), srcOp.getNumLinksAttr(),
      srcOp.getTopologyAttr(), muxCoreRangeSet,
      srcOp.getOutputMemoryConfigAttr());

  rewriter.replaceOp(srcOp, newOp.getResults());
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
