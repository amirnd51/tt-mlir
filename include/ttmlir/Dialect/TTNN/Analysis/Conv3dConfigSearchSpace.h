// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_CONV3DCONFIGSEARCHSPACE_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_CONV3DCONFIGSEARCHSPACE_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/ADT/SmallVector.h"

#include <functional>
#include <optional>

namespace mlir {
namespace tt {
namespace ttnn {

// Search space for ttnn::Conv3dConfigAttr. Each non-empty SmallVector defines
// the candidate set for that field; an empty vector means the field is fixed
// to whatever the base config or default has.
//
// Unlike Conv2dConfigSearchSpace, every field except weightsDtype is a uint32
// block size. The cardinality of the cartesian product is dominated by the
// block-size dimensions; Conv3dConfigGenerator uses divisibility / utilization
// filters to keep the realized candidate set tractable.
struct Conv3dConfigSearchSpace {
  llvm::SmallVector<ttcore::DataType> weightsDtype;
  llvm::SmallVector<uint32_t> tOutBlock;
  llvm::SmallVector<uint32_t> wOutBlock;
  llvm::SmallVector<uint32_t> hOutBlock;
  llvm::SmallVector<uint32_t> cOutBlock;
  llvm::SmallVector<uint32_t> cInBlock;
  llvm::SmallVector<::mlir::tt::ttcore::GridAttr> computeWithStorageGridSize;

  Conv3dConfigSearchSpace() = default;

  bool isWeightsDtypeSetForSearch() const { return !weightsDtype.empty(); }
  bool isTOutBlockSetForSearch() const { return !tOutBlock.empty(); }
  bool isWOutBlockSetForSearch() const { return !wOutBlock.empty(); }
  bool isHOutBlockSetForSearch() const { return !hOutBlock.empty(); }
  bool isCOutBlockSetForSearch() const { return !cOutBlock.empty(); }
  bool isCInBlockSetForSearch() const { return !cInBlock.empty(); }
  bool isComputeWithStorageGridSizeSetForSearch() const {
    return !computeWithStorageGridSize.empty();
  }

  bool isAnyFieldSetForSearch() const {
    return isWeightsDtypeSetForSearch() || isTOutBlockSetForSearch() ||
           isWOutBlockSetForSearch() || isHOutBlockSetForSearch() ||
           isCOutBlockSetForSearch() || isCInBlockSetForSearch() ||
           isComputeWithStorageGridSizeSetForSearch();
  }
};

// Eager cartesian-product generator over Conv3dConfigSearchSpace. Mirrors
// Conv2dConfigGenerator but constructs Conv3dConfigAttr fresh on each
// iteration (Conv3dConfigAttr has no fluent with*-setters).
//
// Use:
//   Conv3dConfigGenerator gen(op, baseConfig, space, filterOutFn);
//   while (auto attr = gen.getNextConfig()) { ... }
//
// filterOutFn receives the produced Conv3dConfigAttr and returns true to
// reject it. Use this for empirical legality rules — e.g. divisibility,
// h_out_block * w_out_block <= 256, minimum core utilization.
class Conv3dConfigGenerator {
public:
  Conv3dConfigGenerator(
      Conv3dOp *op, Conv3dConfigAttr baseConfig,
      const Conv3dConfigSearchSpace &space,
      std::function<bool(const Conv3dConfigAttr &)> filterOutFn);

  // Returns the next configuration in the search space, or nullptr when
  // exhausted.
  ::mlir::tt::ttnn::Conv3dConfigAttr getNextConfig();

  // True when getNextConfig will return nullptr on the next call.
  bool searchDone() const { return isDone; }

private:
  // Each active field has a vector of candidate values and a cursor into it.
  // The generator advances cursors like a multi-digit odometer (most
  // significant = first active field).
  struct FieldCursor {
    size_t index = 0;
    size_t size = 0;
    bool advance() {
      if (++index >= size) {
        index = 0;
        return true; // wrapped
      }
      return false;
    }
  };

  // Snapshot of the *currently selected* value for each Conv3dConfigAttr
  // field. baseConfig pre-populates fields not in the search; active search
  // fields overwrite their slot each iteration before the attr is built.
  struct ConfigSnapshot {
    std::optional<ttcore::DataType> weightsDtype;
    std::optional<uint32_t> tOutBlock;
    std::optional<uint32_t> wOutBlock;
    std::optional<uint32_t> hOutBlock;
    std::optional<uint32_t> cOutBlock;
    std::optional<uint32_t> cInBlock;
    std::optional<ttcore::GridAttr> computeWithStorageGridSize;
  };

  enum class Field {
    WeightsDtype,
    TOutBlock,
    WOutBlock,
    HOutBlock,
    COutBlock,
    CInBlock,
    ComputeWithStorageGridSize,
  };

  struct ActiveField {
    Field field;
    FieldCursor cursor;
  };

  Conv3dConfigAttr buildAttrFromSnapshot() const;
  void applyCursorToSnapshot(const ActiveField &af);

  [[maybe_unused]] Conv3dOp *op;
  ::mlir::MLIRContext *ctx;
  Conv3dConfigSearchSpace searchSpace;
  ConfigSnapshot snapshot;
  llvm::SmallVector<ActiveField> activeFields;
  std::function<bool(const Conv3dConfigAttr &)> filterOutFn;
  bool isDone = false;
};

} // namespace ttnn
} // namespace tt
} // namespace mlir

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_CONV3DCONFIGSEARCHSPACE_H
