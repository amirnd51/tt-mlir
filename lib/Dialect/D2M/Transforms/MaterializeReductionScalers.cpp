// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "D2MMaterializeReductionScalers"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MMATERIALIZEREDUCTIONSCALERS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kMaterializedReductionScalerAttr =
    "d2m.materialized_reduction_scaler";

static bool isReductionScalerUse(d2m::TileFillOp fillOp, Operation *user) {
  Value fill = fillOp.getResult();
  return llvm::TypeSwitch<Operation *, bool>(user)
      .Case<d2m::TileReduceSumOp, d2m::TileReduceMaxOp, d2m::TileReduceMeanOp>(
          [&](auto reduceOp) { return reduceOp.getB() == fill; })
      .Default(false);
}

static SmallVector<d2m::TileFillOp>
collectReductionScalerFills(linalg::GenericOp linalgOp) {
  SmallVector<d2m::TileFillOp> fills;
  llvm::SmallPtrSet<Operation *, 4> seen;
  linalgOp.getBody()->walk([&](d2m::TileFillOp fillOp) {
    if (!llvm::any_of(fillOp->getUsers(), [&](Operation *user) {
          return isReductionScalerUse(fillOp, user);
        })) {
      return;
    }
    if (seen.insert(fillOp.getOperation()).second) {
      fills.push_back(fillOp);
    }
  });
  return fills;
}

static FailureOr<Value> cloneConstantScalar(Value value, RewriterBase &rewriter,
                                            IRMapping &mapping) {
  if (mapping.contains(value)) {
    return mapping.lookup(value);
  }

  Operation *definingOp = value.getDefiningOp();
  if (!definingOp || !isa<arith::ConstantOp>(definingOp)) {
    return failure();
  }

  Operation *clonedOp = rewriter.clone(*definingOp, mapping);
  if (clonedOp->getNumResults() != 1) {
    return failure();
  }
  return clonedOp->getResult(0);
}

static void copyNonStructuralAttrs(linalg::GenericOp from,
                                   linalg::GenericOp to) {
  for (NamedAttribute attr : from->getAttrs()) {
    StringRef name = attr.getName().getValue();
    if (name == "indexing_maps" || name == "iterator_types" ||
        name == "operandSegmentSizes") {
      continue;
    }
    to->setAttr(attr.getName(), attr.getValue());
  }
}

static Type getElementType(Value value) {
  return cast<ShapedType>(value.getType()).getElementType();
}

static FailureOr<Value>
createScalerBufferAndProducer(linalg::GenericOp consumer,
                              d2m::TileFillOp fillOp, AffineMap outputMap,
                              RewriterBase &rewriter) {
  Location loc = fillOp.getLoc();
  auto outputType =
      dyn_cast<MemRefType>(consumer.getOutputs().front().getType());
  if (!outputType) {
    return failure();
  }

  rewriter.setInsertionPoint(consumer);
  auto scalerBuffer = rewriter.create<memref::AllocOp>(loc, outputType);
  scalerBuffer->setAttr(kMaterializedReductionScalerAttr,
                        rewriter.getUnitAttr());

  auto producer = rewriter.create<linalg::GenericOp>(
      loc, TypeRange(), ValueRange(), ValueRange{scalerBuffer.getResult()},
      ArrayRef<AffineMap>{outputMap}, consumer.getIteratorTypesArray());

  Block *producerBody = rewriter.createBlock(&producer.getRegion());
  producerBody->addArgument(outputType.getElementType(), loc);

  rewriter.setInsertionPointToStart(producerBody);
  IRMapping scalarMapping;
  FailureOr<Value> fillValue =
      cloneConstantScalar(fillOp.getValue(), rewriter, scalarMapping);
  if (failed(fillValue)) {
    return failure();
  }

  auto materializedFill = rewriter.create<d2m::TileFillOp>(
      loc, fillOp.getResult().getType(), *fillValue);
  rewriter.create<linalg::YieldOp>(loc, materializedFill.getResult());
  return scalerBuffer.getResult();
}

static linalg::GenericOp rebuildConsumer(linalg::GenericOp oldOp,
                                         ArrayRef<d2m::TileFillOp> fills,
                                         ArrayRef<Value> scalerBuffers,
                                         AffineMap outputMap,
                                         RewriterBase &rewriter) {
  SmallVector<Value> newInputs(oldOp.getInputs().begin(),
                               oldOp.getInputs().end());
  llvm::append_range(newInputs, scalerBuffers);

  SmallVector<AffineMap> newIndexingMaps;
  SmallVector<AffineMap> oldIndexingMaps = oldOp.getIndexingMapsArray();
  unsigned numOldInputs = oldOp.getNumDpsInputs();
  for (unsigned i = 0; i < numOldInputs; ++i) {
    newIndexingMaps.push_back(oldIndexingMaps[i]);
  }
  for ([[maybe_unused]] Value scalerBuffer : scalerBuffers) {
    newIndexingMaps.push_back(outputMap);
  }
  for (unsigned i = numOldInputs; i < oldIndexingMaps.size(); ++i) {
    newIndexingMaps.push_back(oldIndexingMaps[i]);
  }

  rewriter.setInsertionPoint(oldOp);
  auto newOp = rewriter.create<linalg::GenericOp>(
      oldOp.getLoc(), oldOp.getResultTypes(), newInputs, oldOp.getOutputs(),
      newIndexingMaps, oldOp.getIteratorTypesArray());
  copyNonStructuralAttrs(oldOp, newOp);

  Block *oldBody = oldOp.getBody();
  Block *newBody = rewriter.createBlock(&newOp.getRegion());

  for (Value input : oldOp.getInputs()) {
    newBody->addArgument(getElementType(input), oldOp.getLoc());
  }
  for (Value scalerBuffer : scalerBuffers) {
    newBody->addArgument(getElementType(scalerBuffer), oldOp.getLoc());
  }
  for (Value output : oldOp.getOutputs()) {
    newBody->addArgument(getElementType(output), oldOp.getLoc());
  }

  IRMapping mapping;
  for (unsigned i = 0; i < numOldInputs; ++i) {
    mapping.map(oldBody->getArgument(i), newBody->getArgument(i));
  }
  for (auto [idx, fillOp] : llvm::enumerate(fills)) {
    d2m::TileFillOp nonConstFillOp = fillOp;
    mapping.map(nonConstFillOp.getResult(),
                newBody->getArgument(numOldInputs + idx));
  }
  unsigned newOutputArgStart = numOldInputs + scalerBuffers.size();
  for (unsigned i = 0; i < oldOp.getNumDpsInits(); ++i) {
    mapping.map(oldBody->getArgument(numOldInputs + i),
                newBody->getArgument(newOutputArgStart + i));
  }

  llvm::SmallPtrSet<Operation *, 4> fillsToSkip;
  for (d2m::TileFillOp fillOp : fills) {
    fillsToSkip.insert(fillOp.getOperation());
  }

  rewriter.setInsertionPointToStart(newBody);
  for (Operation &bodyOp : oldBody->getOperations()) {
    if (fillsToSkip.contains(&bodyOp)) {
      continue;
    }
    rewriter.clone(bodyOp, mapping);
  }

  return newOp;
}

static LogicalResult materializeReductionScalers(linalg::GenericOp linalgOp,
                                                 RewriterBase &rewriter) {
  if (linalgOp.getNumDpsInits() != 1) {
    return failure();
  }

  SmallVector<d2m::TileFillOp> fills = collectReductionScalerFills(linalgOp);
  if (fills.empty()) {
    return failure();
  }

  unsigned outputMapIndex = linalgOp.getNumDpsInputs();
  AffineMap outputMap = linalgOp.getIndexingMapsArray()[outputMapIndex];
  SmallVector<Value> scalerBuffers;
  scalerBuffers.reserve(fills.size());
  for (d2m::TileFillOp fillOp : fills) {
    FailureOr<Value> scalerBuffer =
        createScalerBufferAndProducer(linalgOp, fillOp, outputMap, rewriter);
    if (failed(scalerBuffer)) {
      return failure();
    }
    scalerBuffers.push_back(*scalerBuffer);
  }

  linalg::GenericOp newOp =
      rebuildConsumer(linalgOp, fills, scalerBuffers, outputMap, rewriter);
  if (linalgOp->getNumResults() > 0) {
    rewriter.replaceOp(linalgOp, newOp->getResults());
  } else {
    rewriter.eraseOp(linalgOp);
  }
  return success();
}

class D2MMaterializeReductionScalers
    : public impl::D2MMaterializeReductionScalersBase<
          D2MMaterializeReductionScalers> {
public:
  using impl::D2MMaterializeReductionScalersBase<
      D2MMaterializeReductionScalers>::D2MMaterializeReductionScalersBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    SmallVector<linalg::GenericOp> linalgOps;
    moduleOp.walk([&](linalg::GenericOp linalgOp) {
      if (linalgOp->getParentOfType<d2m::GenericOp>()) {
        linalgOps.push_back(linalgOp);
      }
    });

    for (linalg::GenericOp linalgOp : linalgOps) {
      if (!linalgOp->getBlock()) {
        continue;
      }
      if (failed(materializeReductionScalers(linalgOp, rewriter))) {
        continue;
      }
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
