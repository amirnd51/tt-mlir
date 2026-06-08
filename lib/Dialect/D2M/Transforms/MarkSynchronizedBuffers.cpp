// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/SynchronizableOpInterfaceUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"

#include <cstdlib>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MMARKSYNCHRONIZEDBUFFERS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

static bool containsAccumulatingCompute(Operation *op) {
  if (isa<d2m::TileMatmulOp, d2m::TileMatmulBlockOp, d2m::TileReduceSumOp,
          d2m::TileReduceMaxOp, d2m::TileReduceMeanOp, d2m::TileSFPUReduceSumOp,
          d2m::TileSFPUReduceMaxOp>(op)) {
    return true;
  }

  return op
      ->walk([](Operation *nestedOp) {
        if (isa<d2m::TileMatmulOp, d2m::TileMatmulBlockOp, d2m::TileReduceSumOp,
                d2m::TileReduceMaxOp, d2m::TileReduceMeanOp,
                d2m::TileSFPUReduceSumOp, d2m::TileSFPUReduceMaxOp>(nestedOp)) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      })
      .wasInterrupted();
}

class D2MMarkSynchronizedBuffers
    : public impl::D2MMarkSynchronizedBuffersBase<D2MMarkSynchronizedBuffers> {
public:
  using impl::D2MMarkSynchronizedBuffersBase<
      D2MMarkSynchronizedBuffers>::D2MMarkSynchronizedBuffersBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    moduleOp->walk([&](d2m::GenericOp genericOp) {
      auto cbUsageInfo = utils::getCBUsageInfo(genericOp.getRegion(0));
      for (auto &[cb, usageInfo] : cbUsageInfo) {
        if (auto allocOp =
                mlir::dyn_cast<memref::AllocOp>(cb.getDefiningOp())) {
          // MOLA opt #3 (deeper software pipelining): override the CB prefetch
          // depth (default 2 = double-buffered) with MOLA_TT_CB_DEPTH so the
          // reader can prefetch N-1 tiles ahead, hiding more DRAM latency on
          // latency-bound matmuls. Accumulators still pin to 1 (can't double-
          // buffer a reduction accumulator).
          static const int32_t depthOverride = [] {
            const char *e = ::getenv("MOLA_TT_CB_DEPTH");
            return e && atoi(e) > 0 ? atoi(e) : 0;
          }();
          int32_t bufferCount =
              depthOverride > 0 ? depthOverride
                                : static_cast<int32_t>(numStreamBuffers);
          for (Operation *producer : usageInfo.producers) {
            if (containsAccumulatingCompute(producer)) {
              bufferCount = 1;
              break;
            }
          }
          allocOp->setAttr("d2m.synchronized_buffer",
                           rewriter.getI32IntegerAttr(bufferCount));

          if (usageInfo.consumers.size() == 1 &&
              usageInfo.producers.size() == 1) {
            auto *consumer = usageInfo.consumers.front();
            auto *producer = usageInfo.producers.front();
            if (mlir::isa<linalg::GenericOp>(consumer) &&
                mlir::isa<linalg::GenericOp>(producer)) {
              allocOp->setAttr("d2m.compute_intermediate",
                               rewriter.getUnitAttr());
            }
          }
        }
      }
    });
  }
};

} // namespace
} // namespace mlir::tt::d2m
