// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/AffineMapAnalysis.h"
#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <optional>
#include <type_traits>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERDMATOFULLYINDEXEDFORM
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

struct CoalescingCacheKey {
  const void *map = nullptr;
  SmallVector<int64_t, 8> shape;
  unsigned numGridDims = 0;
  size_t elemSizeBytes = 0;

  bool operator==(const CoalescingCacheKey &other) const {
    return map == other.map && shape == other.shape &&
           numGridDims == other.numGridDims &&
           elemSizeBytes == other.elemSizeBytes;
  }
};

struct CoalescingCacheKeyInfo {
  static CoalescingCacheKey getEmptyKey() {
    return CoalescingCacheKey{
        DenseMapInfo<const void *>::getEmptyKey(), {}, 0, 0};
  }

  static CoalescingCacheKey getTombstoneKey() {
    return CoalescingCacheKey{
        DenseMapInfo<const void *>::getTombstoneKey(), {}, 0, 0};
  }

  static unsigned getHashValue(const CoalescingCacheKey &key) {
    return static_cast<unsigned>(llvm::hash_combine(
        key.map, key.numGridDims, key.elemSizeBytes,
        llvm::hash_combine_range(key.shape.begin(), key.shape.end())));
  }

  static bool isEqual(const CoalescingCacheKey &lhs,
                      const CoalescingCacheKey &rhs) {
    return lhs == rhs;
  }
};

using CoalescingCache =
    DenseMap<CoalescingCacheKey, size_t, CoalescingCacheKeyInfo>;

class CoalescingFactorCache {
public:
  size_t get(AffineMap memoryMap, ArrayRef<int64_t> gridShape,
             ArrayRef<int64_t> shardShape, size_t elemSizeBytes,
             bool debugCoalescingInference);

private:
  CoalescingCache cache;
};

static size_t getElementSizeBytes(MemRefType memref) {
  mlir::Type elementType = memref.getElementType();
  auto tileType = mlir::dyn_cast<ttcore::TileType>(elementType);
  return tileType ? tileType.getSizeBytes()
                  : elementType.getIntOrFloatBitWidth() / 8;
}

static std::optional<AffineMap> buildTensorAccessorPageMap(Value remoteMemref) {
  MemRefType remoteType = mlir::cast<MemRefType>(remoteMemref.getType());
  MemRefType baseType = remoteType;

  AffineMap viewMap = AffineMap::getMultiDimIdentityMap(
      remoteType.getRank(), remoteType.getContext());
  if (Operation *definingOp = remoteMemref.getDefiningOp()) {
    if (auto viewOp = mlir::dyn_cast<ViewOpInterface>(definingOp)) {
      if (viewOp.isComposite()) {
        return std::nullopt;
      }
      std::tie(baseType, viewMap) = applyViews(definingOp);
    }
  }

  if (!baseType.hasStaticShape() ||
      !mlir::isa<ttcore::TileType>(baseType.getElementType())) {
    return std::nullopt;
  }

  auto baseLayout = mlir::dyn_cast_if_present<ttcore::DeviceLayoutInterface>(
      baseType.getLayout());
  if (!baseLayout ||
      !mlir::isa<ttcore::ShardLayoutAttr, ttcore::InterleavedLayoutAttr>(
          baseType.getLayout())) {
    return std::nullopt;
  }

  ttcore::MemorySpace memorySpace = ttcore::getMemorySpace(baseType);
  if (memorySpace != ttcore::MemorySpace::DeviceL1 &&
      memorySpace != ttcore::MemorySpace::DeviceDRAM) {
    return std::nullopt;
  }
  if (mlir::isa<ttcore::InterleavedLayoutAttr>(baseType.getLayout()) &&
      memorySpace != ttcore::MemorySpace::DeviceDRAM) {
    return std::nullopt;
  }

  ArrayRef<int64_t> gridShape = baseLayout.getGridShape(baseType);
  ArrayRef<int64_t> shardShape = baseLayout.getShardShape(baseType);
  const size_t rank = gridShape.size();
  static constexpr size_t kTensorAccessorMaxRank = 8;
  if (rank == 0 || rank != shardShape.size() || rank > kTensorAccessorMaxRank) {
    return std::nullopt;
  }
  if (viewMap.getNumDims() != static_cast<unsigned>(remoteType.getRank()) ||
      viewMap.getNumSymbols() != 0 ||
      viewMap.getNumResults() != static_cast<unsigned>(baseType.getRank())) {
    return std::nullopt;
  }

  SmallVector<int64_t> tensorShape;
  tensorShape.reserve(rank);
  for (auto [gridDim, shardDim] : llvm::zip_equal(gridShape, shardShape)) {
    tensorShape.push_back(gridDim * shardDim);
  }

  SmallVector<int64_t> tensorStrides(tensorShape.size());
  int64_t stride = 1;
  for (int64_t dim = static_cast<int64_t>(rank) - 1; dim >= 0; dim--) {
    tensorStrides[dim] = stride;
    stride *= tensorShape[dim];
  }

  MLIRContext *ctx = remoteType.getContext();
  AffineExpr pageId = getAffineConstantExpr(0, ctx);
  for (unsigned dim = 0; dim < rank; dim++) {
    AffineExpr gridIndex = getAffineDimExpr(dim, ctx);
    AffineExpr shardIndex = getAffineDimExpr(rank + dim, ctx);
    AffineExpr absolutePage = gridIndex * shardShape[dim] + shardIndex;
    pageId = pageId + absolutePage * tensorStrides[dim];
  }

  AffineMap basePageMap = AffineMap::get(baseType.getRank(), 0, pageId, ctx);
  return simplifyAffineMap(basePageMap.compose(viewMap));
}

template <typename DMAOp>
static std::optional<AffineMap> getEligibleTensorAccessorPageMap(DMAOp op) {
  static_assert(std::is_same_v<DMAOp, DMAReadOp> ||
                std::is_same_v<DMAOp, DMAWriteOp>);

  if (!op.isShardLevel()) {
    return std::nullopt;
  }

  Value remoteMemref = nullptr;
  Value localMemref = nullptr;
  ValueRange remoteGridIndices;
  if constexpr (std::is_same_v<DMAOp, DMAReadOp>) {
    if (!op.isSrcRemote()) {
      return std::nullopt;
    }
    remoteMemref = op.getSrc();
    localMemref = op.getDst();
    remoteGridIndices = op.getSrcIndices();
  } else {
    if (!op.isDstRemote() || op.isMcast() || !op.getStartDevice().empty() ||
        !op.getDeviceMcastShape().empty()) {
      return std::nullopt;
    }
    remoteMemref = op.getDst();
    localMemref = op.getSrc();
    remoteGridIndices = op.getDstIndices();
  }

  GenericOp generic = op->template getParentOfType<GenericOp>();
  if (!generic || !llvm::is_contained(generic.getOperands(), remoteMemref)) {
    return std::nullopt;
  }

  auto remoteType = mlir::dyn_cast<MemRefType>(remoteMemref.getType());
  auto localType = mlir::dyn_cast<MemRefType>(localMemref.getType());
  if (!remoteType || !localType || !remoteType.hasStaticShape() ||
      !localType.hasStaticShape() ||
      !mlir::isa<ttcore::TileType>(remoteType.getElementType()) ||
      remoteType.getElementType() != localType.getElementType()) {
    return std::nullopt;
  }

  int64_t pageSize = ttcore::getElementSizeBytes(remoteType.getElementType());
  if (pageSize % utils::getNocAddressAlignmentBytes(
                     op, ttcore::getMemorySpace(remoteType)) !=
          0 ||
      pageSize % utils::getNocAddressAlignmentBytes(
                     op, ttcore::MemorySpace::DeviceL1) !=
          0) {
    return std::nullopt;
  }

  auto remoteLayout = mlir::dyn_cast_if_present<ttcore::DeviceLayoutInterface>(
      remoteType.getLayout());
  if (!remoteLayout ||
      remoteGridIndices.size() !=
          remoteLayout.getGridShape(remoteType).size() ||
      ttmlir::utils::volume(remoteLayout.getShardShape(remoteType)) !=
          localType.getNumElements()) {
    return std::nullopt;
  }

  return buildTensorAccessorPageMap(remoteMemref);
}

// Calculates coalescing factor using analytical method with sampling fallback.
// Mirrors the logic from GenericLowerDMAs::analyzeStream.
static size_t calculateCoalescingFactorWithFallback(
    AffineMap memoryMap, ArrayRef<int64_t> gridShape,
    ArrayRef<int64_t> shardShape, size_t elemSizeBytes,
    bool debugCoalescingInference) {

  static constexpr size_t coalescingFactorSamplingFallbackThreshold = 16;

  // Compute full shape (grid + shard)
  SmallVector<int64_t> fullShape;
  fullShape.append(gridShape.begin(), gridShape.end());
  fullShape.append(shardShape.begin(), shardShape.end());

  // Try analytical method first
  size_t coalescingFactor = ttmlir::utils::computeCoalescingFactorAnalytically(
      memoryMap, fullShape, gridShape.size(), elemSizeBytes);

  // Determine if we should fallback to sampling
  size_t analyticalChunkSize = coalescingFactor * elemSizeBytes;
  bool shouldFallbackToSampling =
      analyticalChunkSize <= coalescingFactorSamplingFallbackThreshold;

  if (shouldFallbackToSampling || debugCoalescingInference) {
    if (shouldFallbackToSampling) {
      llvm::dbgs() << "Analytical coalescing factor below threshold, "
                      "falling back to sampling based coalescing factor...\n";
    } else {
      llvm::dbgs() << "--------------------------[CoalescingFactor]------------"
                      "--------------------\n";
      llvm::dbgs() << "Computing sampling based coalescing factor...\n";
    }

    size_t sampledCoalescingFactor = ttmlir::utils::calculateCoalescingFactor(
        memoryMap, fullShape, elemSizeBytes, gridShape.size());

    if (debugCoalescingInference) {
      if (coalescingFactor == sampledCoalescingFactor) {
        llvm::dbgs() << "  [✓] Analytical and sampled coalescing "
                        "factors MATCH = "
                     << coalescingFactor << "\n";
      } else if (coalescingFactor != sampledCoalescingFactor &&
                 sampledCoalescingFactor % coalescingFactor == 0) {
        llvm::dbgs() << "  [✓] Analytical coalescing factor is valid, but "
                        "smaller than the sampled coalescing factor!\n";
        llvm::dbgs() << "    analytical = " << coalescingFactor
                     << " vs sampled = " << sampledCoalescingFactor << "\n";
        llvm::dbgs() << "    Map: " << memoryMap << "\n";
        llvm::dbgs() << "    Shape: "
                     << ttmlir::utils::formatIterable(fullShape, "x") << "\n";
        llvm::dbgs() << "  Setting coalescing factor to fallback sampled "
                        "value = "
                     << sampledCoalescingFactor << "\n";
        coalescingFactor = sampledCoalescingFactor;
      }

      if (sampledCoalescingFactor % coalescingFactor != 0) {
        llvm::dbgs() << "  [ERROR] Analytical coalescing factor is not a "
                        "divisor of sampled coalescing factor! Generated DMA "
                        "indexing is likely incorrect!\n";
        llvm::dbgs() << "    Sampled coalescing factor: "
                     << sampledCoalescingFactor << "\n";
        llvm::dbgs() << "    Analytical coalescing factor: " << coalescingFactor
                     << "\n";
        llvm::dbgs() << "    Map: " << memoryMap << "\n";
        llvm::dbgs() << "    Shape: "
                     << ttmlir::utils::formatIterable(fullShape, "x") << "\n";
        llvm::dbgs() << "  Setting coalescing factor to fallback sampled "
                        "value = "
                     << sampledCoalescingFactor << "\n";
        coalescingFactor = sampledCoalescingFactor;
      }
    } else if (shouldFallbackToSampling) {
      // Not in debug mode but we need to fallback
      coalescingFactor = sampledCoalescingFactor;
    }

    llvm::dbgs() << "--------------------------------------------------------"
                    "-------------\n";
  }

  return coalescingFactor;
}

size_t CoalescingFactorCache::get(AffineMap memoryMap,
                                  ArrayRef<int64_t> gridShape,
                                  ArrayRef<int64_t> shardShape,
                                  size_t elemSizeBytes,
                                  bool debugCoalescingInference) {
  CoalescingCacheKey key;
  key.map = memoryMap.getAsOpaquePointer();
  key.shape.append(gridShape.begin(), gridShape.end());
  key.shape.append(shardShape.begin(), shardShape.end());
  key.numGridDims = gridShape.size();
  key.elemSizeBytes = elemSizeBytes;

  auto [it, inserted] = cache.try_emplace(std::move(key));
  if (inserted) {
    it->second = calculateCoalescingFactorWithFallback(
        memoryMap, gridShape, shardShape, elemSizeBytes,
        debugCoalescingInference);
  }
  return it->second;
}

// Core loop generation with coalescing guard.  Handles both the fully
// contiguous case (single call at zero indices) and the strided case
// (nested scf.for loops with an `if (flat_index % CF == 0)` guard).
// The caller supplies `emitDMA` which builds the actual DMA op for a
// given set of iteration indices.
using EmitDMAFn =
    llvm::function_ref<Value(OpBuilder &builder, Location loc, ValueRange iters,
                             size_t coalescingFactor)>;

static Value generateDMAWithCoalescing(OpBuilder &builder, Location loc,
                                       ArrayRef<int64_t> iterShape,
                                       size_t coalescingFactor, DMAType txType,
                                       EmitDMAFn emitDMA) {
  size_t volume = ttmlir::utils::volume(iterShape);

  if (coalescingFactor == volume) {
    Value zero = builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                                   builder.getIndexAttr(0));
    SmallVector<Value> zeros(iterShape.size(), zero);
    return emitDMA(builder, loc, zeros, coalescingFactor);
  }

  auto [lbs, ubs, steps] = utils::getLoopBounds(builder, loc, iterShape);
  auto nullDmaTx = builder.create<NullTxOp>(loc, txType);

  scf::LoopNest loopNest = scf::buildLoopNest(
      builder, loc, lbs, ubs, steps, ValueRange(nullDmaTx),
      [&](OpBuilder &loopBuilder, Location innerLoc, ValueRange iters,
          ValueRange args) {
        Value cfExpr = loopBuilder.create<arith::ConstantOp>(
            innerLoc, loopBuilder.getIndexType(),
            loopBuilder.getIndexAttr(coalescingFactor));
        Value zero = loopBuilder.create<arith::ConstantOp>(
            innerLoc, loopBuilder.getIndexType(),
            loopBuilder.getIntegerAttr(loopBuilder.getIndexType(), 0));

        auto totalIterCount = zero;
        size_t currStride = 1;
        for (int i = iters.size() - 1; i >= 0; i--) {
          Value currStrideExpr = loopBuilder.create<arith::ConstantOp>(
              innerLoc, loopBuilder.getIndexType(),
              loopBuilder.getIndexAttr(currStride));
          auto scaledCount =
              loopBuilder
                  .create<arith::MulIOp>(innerLoc, currStrideExpr, iters[i])
                  .getResult();
          totalIterCount =
              loopBuilder
                  .create<arith::AddIOp>(innerLoc, scaledCount, totalIterCount)
                  .getResult();
          currStride *= iterShape[i];
        }
        auto moduloIterCount =
            loopBuilder.create<arith::RemSIOp>(innerLoc, totalIterCount, cfExpr)
                .getResult();
        auto predicate = loopBuilder.create<arith::CmpIOp>(
            innerLoc, arith::CmpIPredicate::eq, moduloIterCount, zero);

        auto nulltx = loopBuilder.create<NullTxOp>(innerLoc, txType);

        auto ifExpr = loopBuilder.create<scf::IfOp>(
            innerLoc, TypeRange(SmallVector<Value>{nulltx}), predicate,
            true /*addThenBlock*/, true /*addElseBlock*/);

        auto thenBuilder = ifExpr.getThenBodyBuilder();
        Value dmaTx = emitDMA(thenBuilder, innerLoc, iters, coalescingFactor);
        thenBuilder.create<scf::YieldOp>(innerLoc, dmaTx);

        auto elseBuilder = ifExpr.getElseBodyBuilder();
        elseBuilder.create<scf::YieldOp>(innerLoc, args[0]);

        return SmallVector<Value>{ifExpr.getResult(0)};
      });

  return loopNest.results.front();
}

// Tutticore (#1726): a DMA read through a view that maps part of its iteration
// space OUTSIDE its source memref's shape. A padding relayout lowered through
// shared logical space (LowerToLayout's buildLayoutTransformMap) does this:
// the padded tiles of the view have no source tile, and the source's memory
// map would send them to a shard the source does not have (another DRAM
// bank, memory no buffer owns). Such a read is lowered only where its index
// lies inside the source; the rest of the destination is padding, filled
// with zeros (the dma_read carries kZeroFillAttr; D2MToTTKernel reads the
// core's MEM_ZEROS region for it). A view that stays inside its source is
// lowered exactly as before.
static constexpr llvm::StringLiteral kZeroFillAttr = "d2m.zero_fill";
static constexpr llvm::StringLiteral kViewPadsAttr = "d2m.view_pads";
static constexpr int64_t kOutsideSourceScanLimit = int64_t{1} << 22;

// Asked only of a view LowerToLayout marked kViewPadsAttr (a padding
// change that grows the tensor: an index outside the source's shape, in any
// dimension, is padding). Whether `viewMap` (view index -> source index)
// sends any point of grid x shard outside `sourceShape`; and whether each
// coalesced chunk of `coalescingFactor` consecutive shard points is wholly
// inside or wholly outside (else the factor is cut to 1, so a guarded read
// never straddles). `below` / `above`: per source dimension, whether some
// point falls below 0 or at or past its extent (the guard compares only
// those). (Other views are not asked: a view may overflow a shard dimension
// on purpose and stay inside the shard, as a broadcast does.)
static bool viewReadsOutsideSource(AffineMap viewMap,
                                   ArrayRef<int64_t> sourceShape,
                                   ArrayRef<int64_t> gridShape,
                                   ArrayRef<int64_t> shardShape,
                                   size_t &coalescingFactor,
                                   SmallVectorImpl<bool> &below,
                                   SmallVectorImpl<bool> &above,
                                   bool &sameOnEveryCore) {
  const size_t rank = gridShape.size() + shardShape.size();
  if (viewMap.getNumSymbols() != 0 || viewMap.getNumDims() != rank ||
      viewMap.getNumResults() != sourceShape.size()) {
    return false;
  }
  SmallVector<int64_t> dims(gridShape.begin(), gridShape.end());
  dims.append(shardShape.begin(), shardShape.end());
  below.assign(sourceShape.size(), false);
  above.assign(sourceShape.size(), false);
  const int64_t gridVolume = ttmlir::utils::volume(gridShape);
  const int64_t shardVolume = ttmlir::utils::volume(shardShape);
  if (gridVolume <= 0 || shardVolume <= 0 ||
      gridVolume > kOutsideSourceScanLimit / shardVolume) {
    return false;
  }
  auto inside = [&](int64_t g, int64_t t) {
    SmallVector<int64_t> point(rank, 0);
    int64_t rest = t;
    for (size_t i = shardShape.size(); i-- > 0;) {
      point[gridShape.size() + i] = rest % shardShape[i];
      rest /= shardShape[i];
    }
    rest = g;
    for (size_t i = gridShape.size(); i-- > 0;) {
      point[i] = rest % gridShape[i];
      rest /= gridShape[i];
    }
    SmallVector<int64_t> source = viewMap.compose(point);
    bool in = true;
    for (size_t i = 0; i < source.size(); ++i) {
      if (source[i] < 0) {
        below[i] = true;
        in = false;
      }
      if (source[i] >= sourceShape[i]) {
        above[i] = true;
        in = false;
      }
    }
    return in;
  };
  bool any = false;
  bool straddles = false;
  sameOnEveryCore = true;
  const int64_t cf = static_cast<int64_t>(coalescingFactor);
  for (int64_t g = 0; g < gridVolume; ++g) {
    for (int64_t t = 0; t < shardVolume; ++t) {
      const bool in = inside(g, t);
      any = any || !in;
      if (g != 0 && in != inside(0, t)) {
        sameOnEveryCore = false;
      }
      if (cf > 1 && t % cf != 0 && in != inside(g, t - t % cf)) {
        straddles = true;
      }
    }
  }
  if (any && straddles) {
    coalescingFactor = 1;
  }
  return any;
}

namespace {
class D2MLowerDMAReadToFullyIndexed : public OpRewritePattern<DMAReadOp> {
public:
  D2MLowerDMAReadToFullyIndexed(MLIRContext *context,
                                bool debugCoalescingInference,
                                CoalescingFactorCache &coalescingCache)
      : OpRewritePattern<DMAReadOp>(context),
        debugCoalescingInference(debugCoalescingInference),
        coalescingCache(&coalescingCache) {}

private:
  bool debugCoalescingInference;
  CoalescingFactorCache *coalescingCache = nullptr;

public:
  LogicalResult matchAndRewrite(DMAReadOp op,
                                PatternRewriter &rewriter) const final {
    if (op.isFullyIndexed()) {
      return failure();
    }
    if (op.getTensorAccessorPageMapAttr()) {
      return failure();
    }

    Location loc = op.getLoc();
    Value remoteMemref = op.getSrc();
    Value localMemref = op.getDst();

    MemRefType remoteMemrefType = op.getSrcMemRefType();

    ttcore::DeviceLayoutInterface deviceLayout =
        ttcore::getDeviceLayout(remoteMemref);
    if (!deviceLayout) {
      return rewriter.notifyMatchFailure(
          op, "remote memref must have a device layout");
    }

    ShapedType remoteShapedType =
        mlir::cast<ShapedType>(remoteMemref.getType());
    ArrayRef<int64_t> gridShape = deviceLayout.getGridShape(remoteShapedType);
    ArrayRef<int64_t> shardShape = deviceLayout.getShardShape(remoteShapedType);

    ttcore::DeviceAttr device = ttcore::lookupDevice(op);
    AffineMap remoteMemoryMap = utils::getMemoryMap(device, remoteMemref, true);
    AffineMap localMemoryMap = utils::getMemoryMap(device, localMemref, false);

    size_t elemSizeBytes = getElementSizeBytes(remoteMemrefType);
    size_t coalescingFactor =
        coalescingCache->get(remoteMemoryMap, gridShape, shardShape,
                             elemSizeBytes, debugCoalescingInference);

    SmallVector<Value> gridIndices(op.getSrcIndices());

    // (Tutticore #1726: a view that reads outside its source)
    AffineMap viewMap;
    SmallVector<int64_t> sourceShape;
    bool guardSource = false;
    SmallVector<bool> below, above;
    bool sameOnEveryCore = false;
    if (Operation *definingOp = remoteMemref.getDefiningOp()) {
      if (auto viewOp = mlir::dyn_cast<ViewOpInterface>(definingOp);
          viewOp && !viewOp.isComposite() &&
          definingOp->hasAttr(kViewPadsAttr)) {
        MemRefType baseType;
        std::tie(baseType, viewMap) = applyViews(definingOp);
        sourceShape.assign(baseType.getShape().begin(),
                           baseType.getShape().end());
        guardSource = viewReadsOutsideSource(
            viewMap, sourceShape, gridShape, shardShape, coalescingFactor,
            below, above, sameOnEveryCore);
      }
    }

    Value newTx = generateDMAWithCoalescing(
        rewriter, loc, shardShape, coalescingFactor, DMAType::Read,
        [&](OpBuilder &b, Location l, ValueRange iters, size_t cf) -> Value {
          SmallVector<Value> viewIndices =
              llvm::to_vector(llvm::concat<Value>(gridIndices, iters));
          SmallVector<Value> localIndices = llvm::to_vector(iters);

          SmallVector<Value> remoteIndices =
              utils::applyMap(b, l, remoteMemoryMap, viewIndices, true);
          localIndices =
              utils::applyMap(b, l, localMemoryMap, localIndices, false);

          if (!guardSource) {
            return b.create<DMAReadOp>(l, remoteMemref, remoteIndices,
                                       localMemref, localIndices,
                                       b.getI64IntegerAttr(cf));
          }
          // In the source: 0 <= viewMap(index)[i] < sourceShape[i], for each
          // bound the scan found crossed. Each comparison becomes an index 0
          // or 1 (select) and their product is compared once, so every i1 is
          // a comparison used only as a condition.
          Value one = b.create<arith::ConstantIndexOp>(l, 1);
          Value zero = b.create<arith::ConstantIndexOp>(l, 0);
          Value product = one;
          auto factor = [&](arith::CmpIPredicate pred, Value lhs,
                            int64_t rhs) {
            Value c = b.create<arith::CmpIOp>(
                l, pred, lhs, b.create<arith::ConstantIndexOp>(l, rhs));
            Value bit = b.create<arith::SelectOp>(l, c, one, zero);
            product = b.create<arith::MulIOp>(l, product, bit);
          };
          // Where every core reads the same tiles of the source in bounds,
          // the guard is asked at grid point 0: a function of the loop index
          // alone, the same on every core.
          SmallVector<Value> guardIndices = viewIndices;
          if (sameOnEveryCore) {
            for (size_t g = 0; g < gridIndices.size(); ++g) {
              guardIndices[g] = zero;
            }
          }
          for (unsigned i = 0; i < viewMap.getNumResults(); ++i) {
            if (!below[i] && !above[i]) {
              continue;
            }
            Value at = b.create<affine::AffineApplyOp>(
                l, viewMap.getSubMap({i}), guardIndices);
            if (below[i]) {
              factor(arith::CmpIPredicate::sge, at, 0);
            }
            if (above[i]) {
              factor(arith::CmpIPredicate::slt, at, sourceShape[i]);
            }
          }
          Value inside = b.create<arith::CmpIOp>(
              l, arith::CmpIPredicate::eq, product, one);
          Type txType = op.getResult().getType();
          auto branch = b.create<scf::IfOp>(l, TypeRange{txType}, inside,
                                            /*addThenBlock=*/true,
                                            /*addElseBlock=*/true);
          {
            OpBuilder then = branch.getThenBodyBuilder();
            Value tx = then.create<DMAReadOp>(l, remoteMemref, remoteIndices,
                                              localMemref, localIndices,
                                              then.getI64IntegerAttr(cf));
            then.create<scf::YieldOp>(l, tx);
          }
          {
            // Padding: the same destination, filled with zeros. (The source
            // indices are kept for the op's form and are not read.)
            OpBuilder other = branch.getElseBodyBuilder();
            auto fill = other.create<DMAReadOp>(l, remoteMemref, remoteIndices,
                                                localMemref, localIndices,
                                                other.getI64IntegerAttr(cf));
            fill->setAttr(kZeroFillAttr, other.getUnitAttr());
            other.create<scf::YieldOp>(l, fill.getResult());
          }
          return branch.getResult(0);
        });

    rewriter.replaceOp(op, newTx);
    return success();
  }
};

class D2MLowerDMAWriteToFullyIndexed : public OpRewritePattern<DMAWriteOp> {
public:
  D2MLowerDMAWriteToFullyIndexed(MLIRContext *context,
                                 bool debugCoalescingInference,
                                 CoalescingFactorCache &coalescingCache)
      : OpRewritePattern<DMAWriteOp>(context),
        debugCoalescingInference(debugCoalescingInference),
        coalescingCache(&coalescingCache) {}

private:
  bool debugCoalescingInference;
  CoalescingFactorCache *coalescingCache = nullptr;

public:
  LogicalResult matchAndRewrite(DMAWriteOp op,
                                PatternRewriter &rewriter) const final {
    if (op.isFullyIndexed()) {
      return failure();
    }
    if (op.getTensorAccessorPageMapAttr()) {
      return failure();
    }

    Location loc = op.getLoc();
    Value localMemref = op.getSrc();
    Value dstMemref = op.getDst();

    ttcore::DeviceAttr device = ttcore::lookupDevice(op);

    if (op.isMcast()) {
      // Mcast write: local-to-local, compute local memory map and apply to
      // zero indices to get the fully-indexed form.
      AffineMap localMemoryMap =
          utils::getMemoryMap(device, localMemref, false);

      MemRefType localType = op.getSrcMemRefType();
      ArrayRef<int64_t> shardShape = localType.getShape();
      size_t shardVolume = ttmlir::utils::volume(shardShape);

      SmallVector<Value> localIndices;
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(), rewriter.getIndexAttr(0));
      for (size_t i = 0; i < shardShape.size(); ++i) {
        localIndices.push_back(zero);
      }
      localIndices =
          utils::applyMap(rewriter, loc, localMemoryMap, localIndices, false);

      Value newTx = rewriter.create<DMAWriteOp>(
          loc, localMemref, localIndices, dstMemref, localIndices,
          op.getMcastStartIndex(), op.getMcastShape(), shardVolume);
      rewriter.replaceOp(op, newTx);
      return success();
    }

    // Non-mcast write: local src to remote dst.
    MemRefType remoteMemrefType = op.getDstMemRefType();

    ttcore::DeviceLayoutInterface deviceLayout =
        ttcore::getDeviceLayout(dstMemref);
    if (!deviceLayout) {
      return rewriter.notifyMatchFailure(
          op, "remote memref must have a device layout");
    }

    ShapedType remoteShapedType = mlir::cast<ShapedType>(dstMemref.getType());
    ArrayRef<int64_t> gridShape = deviceLayout.getGridShape(remoteShapedType);
    ArrayRef<int64_t> shardShape = deviceLayout.getShardShape(remoteShapedType);

    AffineMap remoteMemoryMap = utils::getMemoryMap(device, dstMemref, true);
    AffineMap localMemoryMap = utils::getMemoryMap(device, localMemref, false);

    size_t elemSizeBytes = getElementSizeBytes(remoteMemrefType);
    size_t coalescingFactor =
        coalescingCache->get(remoteMemoryMap, gridShape, shardShape,
                             elemSizeBytes, debugCoalescingInference);

    SmallVector<Value> gridIndices(op.getDstIndices());
    SmallVector<Value> startDevice(op.getStartDevice());
    SmallVector<Value> deviceMcastShape(op.getDeviceMcastShape());

    Value newTx = generateDMAWithCoalescing(
        rewriter, loc, shardShape, coalescingFactor, DMAType::Write,
        [&](OpBuilder &b, Location l, ValueRange iters, size_t cf) -> Value {
          SmallVector<Value> remoteIndices =
              llvm::to_vector(llvm::concat<Value>(gridIndices, iters));
          SmallVector<Value> localIndices = llvm::to_vector(iters);

          remoteIndices =
              utils::applyMap(b, l, remoteMemoryMap, remoteIndices, true);
          localIndices =
              utils::applyMap(b, l, localMemoryMap, localIndices, false);

          return b.create<DMAWriteOp>(l, localMemref, localIndices, dstMemref,
                                      remoteIndices, cf, startDevice,
                                      deviceMcastShape);
        });

    rewriter.replaceOp(op, newTx);
    return success();
  }
};

class D2MLowerLocalCopyToFullyIndexed : public OpRewritePattern<LocalCopyOp> {
public:
  D2MLowerLocalCopyToFullyIndexed(MLIRContext *context,
                                  bool debugCoalescingInference)
      : OpRewritePattern<LocalCopyOp>(context),
        debugCoalescingInference(debugCoalescingInference) {}

private:
  bool debugCoalescingInference;

public:
  LogicalResult matchAndRewrite(LocalCopyOp op,
                                PatternRewriter &rewriter) const final {
    if (!op.hasMemTxResult()) {
      return failure();
    }

    Location loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getDst();

    MemRefType srcType;
    if (auto cbType = mlir::dyn_cast<CBType>(src.getType())) {
      srcType = cbType.getUnderlyingAs<MemRefType>();
    } else {
      srcType = mlir::cast<MemRefType>(src.getType());
    }
    MemRefType dstType;
    if (auto cbType = mlir::dyn_cast<CBType>(dst.getType())) {
      dstType = cbType.getUnderlyingAs<MemRefType>();
    } else {
      dstType = mlir::cast<MemRefType>(dst.getType());
    }

    ArrayRef<int64_t> dstShape = dstType.getShape();
    size_t elemSizeBytes = getElementSizeBytes(srcType);

    ArrayRef<Attribute> maps = op.getIndexingMaps().getValue();
    AffineMap srcIndexingMap = mlir::cast<AffineMapAttr>(maps[0]).getValue();
    AffineMap dstIndexingMap = mlir::cast<AffineMapAttr>(maps[1]).getValue();

    if (!dstIndexingMap.isIdentity()) {
      return rewriter.notifyMatchFailure(
          op, "non-identity dst indexing map not yet supported");
    }

    MLIRContext *ctx = rewriter.getContext();
    AffineMap srcMemoryMap = utils::canonicalStridedMap(
        ctx, srcType.getShape(), srcType.getElementType(),
        AffineMap::getMultiDimIdentityMap(srcType.getRank(), ctx));
    AffineMap dstMemoryMap = utils::canonicalStridedMap(
        ctx, dstType.getShape(), dstType.getElementType(),
        AffineMap::getMultiDimIdentityMap(dstType.getRank(), ctx));

    AffineMap composedSrcMap = srcMemoryMap.compose(srcIndexingMap);
    AffineMap composedDstMap = dstMemoryMap.compose(dstIndexingMap);

    // Pad the composed source map with dummy grid dims (all size 1) so that
    // the sampling loop executes at least once.
    // The function requires numGridDims > 0 for correct behaviour.
    unsigned dstRank = dstType.getRank();
    SmallVector<AffineExpr> dimShift;
    for (unsigned i = 0; i < dstRank; ++i) {
      dimShift.push_back(getAffineDimExpr(i + dstRank, ctx));
    }
    AffineMap paddedSrcMap =
        composedSrcMap.replaceDimsAndSymbols(dimShift, {}, 2 * dstRank, 0);
    SmallVector<int64_t> dummyGridShape(dstRank, 1);
    size_t coalescingFactor = calculateCoalescingFactorWithFallback(
        paddedSrcMap, dummyGridShape, dstShape, elemSizeBytes,
        debugCoalescingInference);

    // Collect all enclosing blocking loops (there may be multiple for
    // multi-dimensional blocking, e.g. block_factors = [BF0, BF1]).
    SmallVector<std::pair<Value, int64_t>> blockingLoops;
    for (Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
        if (auto attr =
                forOp->getAttrOfType<IntegerAttr>("d2m.blocking_loop")) {
          blockingLoops.emplace_back(forOp.getInductionVar(), attr.getInt());
        }
      } else if (auto forOp = dyn_cast<affine::AffineForOp>(parent)) {
        if (auto attr =
                forOp->getAttrOfType<IntegerAttr>("d2m.blocking_loop")) {
          blockingLoops.emplace_back(forOp.getInductionVar(), attr.getInt());
        }
      }
    }

    for (auto [blockIV, blockingDim] : blockingLoops) {
      if (static_cast<size_t>(blockingDim) >=
          static_cast<size_t>(dstShape.size())) {
        return rewriter.notifyMatchFailure(
            op, "blocking_loop dimension exceeds dst rank");
      }
    }

    Value newTx = generateDMAWithCoalescing(
        rewriter, loc, dstShape, coalescingFactor, DMAType::Read,
        [&](OpBuilder &b, Location l, ValueRange iters, size_t cf) -> Value {
          SmallVector<Value> srcEvalIters(iters.begin(), iters.end());
          for (auto [blockIV, blockingDim] : blockingLoops) {
            Value blockSize = b.create<arith::ConstantOp>(
                l, b.getIndexType(), b.getIndexAttr(dstShape[blockingDim]));
            Value offset = b.create<arith::MulIOp>(l, blockIV, blockSize);
            srcEvalIters[blockingDim] =
                b.create<arith::AddIOp>(l, offset, iters[blockingDim]);
          }

          SmallVector<Value> srcIndices =
              utils::applyMap(b, l, composedSrcMap, srcEvalIters, false);
          SmallVector<Value> dstIndices =
              utils::applyMap(b, l, composedDstMap, iters, false);

          return b
              .create<DMAReadOp>(l, src, srcIndices, dst, dstIndices,
                                 b.getI64IntegerAttr(cf))
              .getResult();
        });

    rewriter.replaceOp(op, newTx);
    return success();
  }
};

class D2MLowerDMAToFullyIndexedForm
    : public impl::D2MLowerDMAToFullyIndexedFormBase<
          D2MLowerDMAToFullyIndexedForm> {
public:
  using impl::D2MLowerDMAToFullyIndexedFormBase<
      D2MLowerDMAToFullyIndexedForm>::D2MLowerDMAToFullyIndexedFormBase;

  void runOnOperation() final {
    WalkResult walkResult = getOperation()->walk([&](Operation *op) {
      return llvm::TypeSwitch<Operation *, WalkResult>(op)
          .Case<DMAReadOp, DMAWriteOp>([&](auto dmaOp) {
            if (dmaOp.getTensorAccessorPageMapAttr()) {
              dmaOp.emitOpError(
                  "unexpected pre-existing TensorAccessor page map");
              return WalkResult::interrupt();
            }
            if (useTensorAccessorDMA) {
              if (std::optional<AffineMap> pageMap =
                      getEligibleTensorAccessorPageMap(dmaOp)) {
                dmaOp.setTensorAccessorPageMapAttr(
                    AffineMapAttr::get(*pageMap));
              }
            }
            return WalkResult::advance();
          })
          .Default([](Operation *) { return WalkResult::advance(); });
    });
    if (walkResult.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    CoalescingFactorCache coalescingCache;
    RewritePatternSet dmaPatterns(&getContext());
    dmaPatterns.add<D2MLowerDMAReadToFullyIndexed>(
        &getContext(), debugCoalescingInference, coalescingCache);
    dmaPatterns.add<D2MLowerDMAWriteToFullyIndexed>(
        &getContext(), debugCoalescingInference, coalescingCache);
    dmaPatterns.add<D2MLowerLocalCopyToFullyIndexed>(&getContext(),
                                                     debugCoalescingInference);
    walkAndApplyPatterns(getOperation(), std::move(dmaPatterns));
  }
};
} // namespace

} // namespace mlir::tt::d2m
