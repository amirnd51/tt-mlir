// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/Conv3dConfigSearchSpace.h"

#include "ttmlir/Support/Logger.h"

namespace mlir::tt::ttnn {

Conv3dConfigGenerator::Conv3dConfigGenerator(
    Conv3dOp *op, Conv3dConfigAttr baseConfig,
    const Conv3dConfigSearchSpace &space,
    std::function<bool(const Conv3dConfigAttr &)> filterOutFn)
    : op(op), ctx(baseConfig.getContext()), searchSpace(space),
      filterOutFn(std::move(filterOutFn)) {

  assert(baseConfig &&
         "Conv3dConfigGenerator requires a non-null baseConfig so it can "
         "source an MLIRContext.");

  // Seed snapshot from baseConfig: any field already pinned on baseConfig is
  // copied through verbatim and excluded from the search.
  snapshot.weightsDtype = baseConfig.getWeightsDtype();
  snapshot.tOutBlock = baseConfig.getTOutBlock();
  snapshot.wOutBlock = baseConfig.getWOutBlock();
  snapshot.hOutBlock = baseConfig.getHOutBlock();
  snapshot.cOutBlock = baseConfig.getCOutBlock();
  snapshot.cInBlock = baseConfig.getCInBlock();
  snapshot.computeWithStorageGridSize =
      baseConfig.getComputeWithStorageGridSize();

  auto addActive = [&](Field f, size_t n) {
    activeFields.push_back(
        ActiveField{f, FieldCursor{/*index=*/0, /*size=*/n}});
  };

  if (searchSpace.isWeightsDtypeSetForSearch() &&
      !snapshot.weightsDtype.has_value()) {
    addActive(Field::WeightsDtype, searchSpace.weightsDtype.size());
  }
  if (searchSpace.isTOutBlockSetForSearch() &&
      !snapshot.tOutBlock.has_value()) {
    addActive(Field::TOutBlock, searchSpace.tOutBlock.size());
  }
  if (searchSpace.isWOutBlockSetForSearch() &&
      !snapshot.wOutBlock.has_value()) {
    addActive(Field::WOutBlock, searchSpace.wOutBlock.size());
  }
  if (searchSpace.isHOutBlockSetForSearch() &&
      !snapshot.hOutBlock.has_value()) {
    addActive(Field::HOutBlock, searchSpace.hOutBlock.size());
  }
  if (searchSpace.isCOutBlockSetForSearch() &&
      !snapshot.cOutBlock.has_value()) {
    addActive(Field::COutBlock, searchSpace.cOutBlock.size());
  }
  if (searchSpace.isCInBlockSetForSearch() && !snapshot.cInBlock.has_value()) {
    addActive(Field::CInBlock, searchSpace.cInBlock.size());
  }
  if (searchSpace.isComputeWithStorageGridSizeSetForSearch() &&
      !snapshot.computeWithStorageGridSize.has_value()) {
    addActive(Field::ComputeWithStorageGridSize,
              searchSpace.computeWithStorageGridSize.size());
  }

  isDone = activeFields.empty();

  // Initialize the snapshot's active-field slots to the first candidate
  // value so the first call to getNextConfig() returns a well-formed attr.
  for (const ActiveField &af : activeFields) {
    applyCursorToSnapshot(af);
  }
}

void Conv3dConfigGenerator::applyCursorToSnapshot(const ActiveField &af) {
  size_t i = af.cursor.index;
  switch (af.field) {
  case Field::WeightsDtype:
    snapshot.weightsDtype = searchSpace.weightsDtype[i];
    break;
  case Field::TOutBlock:
    snapshot.tOutBlock = searchSpace.tOutBlock[i];
    break;
  case Field::WOutBlock:
    snapshot.wOutBlock = searchSpace.wOutBlock[i];
    break;
  case Field::HOutBlock:
    snapshot.hOutBlock = searchSpace.hOutBlock[i];
    break;
  case Field::COutBlock:
    snapshot.cOutBlock = searchSpace.cOutBlock[i];
    break;
  case Field::CInBlock:
    snapshot.cInBlock = searchSpace.cInBlock[i];
    break;
  case Field::ComputeWithStorageGridSize:
    snapshot.computeWithStorageGridSize =
        searchSpace.computeWithStorageGridSize[i];
    break;
  }
}

Conv3dConfigAttr Conv3dConfigGenerator::buildAttrFromSnapshot() const {
  return Conv3dConfigAttr::get(ctx, snapshot.weightsDtype, snapshot.tOutBlock,
                               snapshot.wOutBlock, snapshot.hOutBlock,
                               snapshot.cOutBlock, snapshot.cInBlock,
                               snapshot.computeWithStorageGridSize);
}

Conv3dConfigAttr Conv3dConfigGenerator::getNextConfig() {
  if (isDone) {
    return nullptr;
  }

  Conv3dConfigAttr generated = buildAttrFromSnapshot();

  // Odometer-advance: try to bump the least significant active field first.
  // If it wraps to 0, carry over to the more significant neighbour.
  int currentFieldToAdvance = static_cast<int>(activeFields.size()) - 1;
  while (currentFieldToAdvance >= 0) {
    ActiveField &af = activeFields[currentFieldToAdvance];
    bool wrapped = af.cursor.advance();
    applyCursorToSnapshot(af);
    if (wrapped) {
      currentFieldToAdvance--;
    } else {
      break;
    }
  }

  if (currentFieldToAdvance < 0) {
    isDone = true;
  }

  TTMLIR_TRACE(ttmlir::LogComponent::Optimizer, "Next conv3d config: {}",
               generated);

  if (filterOutFn && filterOutFn(generated)) {
    TTMLIR_TRACE(ttmlir::LogComponent::Optimizer, "Filtered out {}", generated);
    return getNextConfig();
  }

  return generated;
}

} // namespace mlir::tt::ttnn
