// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpModel/TTNN/TTNNOutputTensorInference.h"

#ifdef TTMLIR_ENABLE_OPMODEL

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/Conversion.h"

#include "llvm/Support/raw_ostream.h"

#include <ttnn/tensor/tensor_spec.hpp>

// Forward declarations of internal helpers from TTNNOpModel.cpp
namespace mlir::tt::ttnn::op_model {
llvm::Expected<::ttnn::TensorSpec> getPrepareConv2dWeightsOpOutputTensorSpec(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    uint32_t in_channels, uint32_t out_channels, uint32_t batch_size,
    uint32_t input_height, uint32_t input_width,
    llvm::ArrayRef<int32_t> kernel_size, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    uint32_t groups, std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig, bool hasBias,
    bool transpose);

llvm::Expected<::ttnn::TensorSpec>
getPrepareMoEComputeW0W1WeightsOpOutputTensorSpec(
    llvm::ArrayRef<int64_t> w0Shape, TTNNLayoutAttr w0Layout,
    llvm::ArrayRef<int64_t> w1Shape, TTNNLayoutAttr w1Layout,
    std::optional<llvm::ArrayRef<int64_t>> bias0Shape,
    std::optional<TTNNLayoutAttr> bias0Layout,
    std::optional<llvm::ArrayRef<int64_t>> bias1Shape,
    std::optional<TTNNLayoutAttr> bias1Layout, uint32_t hiddenSize,
    uint32_t intermediateSize,
    std::optional<MemoryConfigAttr> outputMemoryConfig);

llvm::Expected<::ttnn::TensorSpec>
getPrepareMoEComputeW2WeightsOpOutputTensorSpec(
    llvm::ArrayRef<int64_t> w2Shape, TTNNLayoutAttr w2Layout,
    std::optional<llvm::ArrayRef<int64_t>> bias2Shape,
    std::optional<TTNNLayoutAttr> bias2Layout, uint32_t hiddenSize,
    uint32_t intermediateSize,
    std::optional<MemoryConfigAttr> outputMemoryConfig);
} // namespace mlir::tt::ttnn::op_model

#endif // TTMLIR_ENABLE_OPMODEL

namespace mlir::tt::ttnn::op_model {

mlir::RankedTensorType
getPreparedConv2dWeightsOutputTensor(Conv2dOp *op,
                                     Conv2dConfigAttr conv2dConfig) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto input = op->getInput().getType();
  auto weight = op->getWeight().getType();
  auto inputLayout = mlir::cast<TTNNLayoutAttr>(input.getEncoding());
  auto weightLayout = mlir::cast<TTNNLayoutAttr>(weight.getEncoding());

  llvm::Expected<::ttnn::TensorSpec> outputTensorSpec =
      getPrepareConv2dWeightsOpOutputTensorSpec(
          input.getShape(), inputLayout, weight.getShape(), weightLayout,
          op->getInChannels(), op->getOutChannels(), op->getBatchSize(),
          op->getInputHeight(), op->getInputWidth(), op->getKernelSize(),
          op->getStride(), op->getPadding(), op->getDilation(), op->getGroups(),
          conv2dConfig, op->getConv2dSliceConfigAttr(),
          op->getBias() != nullptr,
          /* transpose */ false);

  if (!outputTensorSpec) {
    llvm::errs() << llvm::toString(outputTensorSpec.takeError());
    assert(false && "Failed to calculate conv2d prepared weights shape.");
  }

  // Convert back to RankedTensorType
  auto deviceGrid =
      ttcore::lookupDevice(op->getOperation()).getWorkerGrid().getShape();

  auto outputLayout = conversion::getLayoutAttrFromTensorSpec(
      op->getContext(), outputTensorSpec.get(), deviceGrid);

  auto shape = outputTensorSpec.get().logical_shape();

  return mlir::RankedTensorType::get(
      llvm::SmallVector<int64_t>(shape.cbegin(), shape.cend()),
      outputLayout.getScalarElementType(), outputLayout);
#else
  assert(false &&
         "Cannot calculate conv2d prepared weights shape without op model");
#endif
}

mlir::RankedTensorType
getPreparedConvTranspose2dWeightsOutputTensor(ConvTranspose2dOp *op,
                                              Conv2dConfigAttr conv2dConfig) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto input = op->getInput().getType();
  auto weight = op->getWeight().getType();
  auto inputLayout = mlir::cast<TTNNLayoutAttr>(input.getEncoding());
  auto weightLayout = mlir::cast<TTNNLayoutAttr>(weight.getEncoding());

  llvm::Expected<::ttnn::TensorSpec> outputTensorSpec =
      getPrepareConv2dWeightsOpOutputTensorSpec(
          input.getShape(), inputLayout, weight.getShape(), weightLayout,
          op->getInChannels(), op->getOutChannels(), op->getBatchSize(),
          op->getInputHeight(), op->getInputWidth(), op->getKernelSize(),
          op->getStride(), op->getPadding(), op->getDilation(), op->getGroups(),
          conv2dConfig, op->getConv2dSliceConfig(), op->getBias() != nullptr,
          /* transpose */ true);

  if (!outputTensorSpec) {
    llvm::errs() << llvm::toString(outputTensorSpec.takeError());
    assert(false &&
           "Failed to calculate conv_transpose2d prepared weights shape.");
  }

  // Convert back to RankedTensorType
  auto deviceGrid =
      ttcore::lookupDevice(op->getOperation()).getWorkerGrid().getShape();

  auto outputLayout = conversion::getLayoutAttrFromTensorSpec(
      op->getContext(), outputTensorSpec.get(), deviceGrid);

  auto shape = outputTensorSpec.get().logical_shape();

  return mlir::RankedTensorType::get(
      llvm::SmallVector<int64_t>(shape.cbegin(), shape.cend()),
      outputLayout.getScalarElementType(), outputLayout);
#else
  assert(false && "Cannot calculate conv_transpose2d prepared weights shape "
                  "without op model");
#endif
}

mlir::RankedTensorType getPreparedMoEComputeW0W1WeightsOutputType(
    mlir::Operation *anchor, mlir::RankedTensorType w0Type,
    mlir::RankedTensorType w1Type,
    std::optional<mlir::RankedTensorType> bias0Type,
    std::optional<mlir::RankedTensorType> bias1Type, uint32_t hiddenSize,
    uint32_t intermediateSize,
    std::optional<MemoryConfigAttr> outputMemoryConfig) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto w0Layout = mlir::cast<TTNNLayoutAttr>(w0Type.getEncoding());
  auto w1Layout = mlir::cast<TTNNLayoutAttr>(w1Type.getEncoding());

  std::optional<llvm::ArrayRef<int64_t>> bias0Shape;
  std::optional<TTNNLayoutAttr> bias0Layout;
  if (bias0Type) {
    bias0Shape = bias0Type->getShape();
    bias0Layout = mlir::cast<TTNNLayoutAttr>(bias0Type->getEncoding());
  }
  std::optional<llvm::ArrayRef<int64_t>> bias1Shape;
  std::optional<TTNNLayoutAttr> bias1Layout;
  if (bias1Type) {
    bias1Shape = bias1Type->getShape();
    bias1Layout = mlir::cast<TTNNLayoutAttr>(bias1Type->getEncoding());
  }

  auto specOrErr = getPrepareMoEComputeW0W1WeightsOpOutputTensorSpec(
      w0Type.getShape(), w0Layout, w1Type.getShape(), w1Layout, bias0Shape,
      bias0Layout, bias1Shape, bias1Layout, hiddenSize, intermediateSize,
      outputMemoryConfig);
  if (!specOrErr) {
    llvm::errs() << llvm::toString(specOrErr.takeError());
    assert(false &&
           "Failed to calculate prepare_moe_compute_w0_w1_weights shape.");
  }

  auto deviceGrid = ttcore::lookupDevice(anchor).getWorkerGrid().getShape();
  auto outLayout = conversion::getLayoutAttrFromTensorSpec(
      anchor->getContext(), specOrErr.get(), deviceGrid);
  auto shape = specOrErr.get().logical_shape();
  return mlir::RankedTensorType::get(
      llvm::SmallVector<int64_t>(shape.cbegin(), shape.cend()),
      outLayout.getScalarElementType(), outLayout);
#else
  assert(false &&
         "Cannot calculate prepare_moe_compute_w0_w1_weights shape without "
         "op model");
#endif
}

mlir::RankedTensorType
getPreparedMoEComputeW0W1WeightsOutputType(PrepareMoEComputeW0W1WeightsOp *op) {
  std::optional<mlir::RankedTensorType> bias0Type;
  if (op->getBias_0()) {
    bias0Type = op->getBias_0().getType();
  }
  std::optional<mlir::RankedTensorType> bias1Type;
  if (op->getBias_1()) {
    bias1Type = op->getBias_1().getType();
  }
  std::optional<MemoryConfigAttr> outputMemoryConfig;
  if (op->getOutputMemoryConfigAttr()) {
    outputMemoryConfig = op->getOutputMemoryConfigAttr();
  }
  return getPreparedMoEComputeW0W1WeightsOutputType(
      op->getOperation(), op->getW0().getType(), op->getW1().getType(),
      bias0Type, bias1Type, op->getHiddenSize(), op->getIntermediateSize(),
      outputMemoryConfig);
}

mlir::RankedTensorType getPreparedMoEComputeW2WeightsOutputType(
    mlir::Operation *anchor, mlir::RankedTensorType w2Type,
    std::optional<mlir::RankedTensorType> bias2Type, uint32_t hiddenSize,
    uint32_t intermediateSize,
    std::optional<MemoryConfigAttr> outputMemoryConfig) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto w2Layout = mlir::cast<TTNNLayoutAttr>(w2Type.getEncoding());

  std::optional<llvm::ArrayRef<int64_t>> bias2Shape;
  std::optional<TTNNLayoutAttr> bias2Layout;
  if (bias2Type) {
    bias2Shape = bias2Type->getShape();
    bias2Layout = mlir::cast<TTNNLayoutAttr>(bias2Type->getEncoding());
  }

  auto specOrErr = getPrepareMoEComputeW2WeightsOpOutputTensorSpec(
      w2Type.getShape(), w2Layout, bias2Shape, bias2Layout, hiddenSize,
      intermediateSize, outputMemoryConfig);
  if (!specOrErr) {
    llvm::errs() << llvm::toString(specOrErr.takeError());
    assert(false &&
           "Failed to calculate prepare_moe_compute_w2_weights shape.");
  }

  auto deviceGrid = ttcore::lookupDevice(anchor).getWorkerGrid().getShape();
  auto outLayout = conversion::getLayoutAttrFromTensorSpec(
      anchor->getContext(), specOrErr.get(), deviceGrid);
  auto shape = specOrErr.get().logical_shape();
  return mlir::RankedTensorType::get(
      llvm::SmallVector<int64_t>(shape.cbegin(), shape.cend()),
      outLayout.getScalarElementType(), outLayout);
#else
  assert(false && "Cannot calculate prepare_moe_compute_w2_weights shape "
                  "without op model");
#endif
}

mlir::RankedTensorType
getPreparedMoEComputeW2WeightsOutputType(PrepareMoEComputeW2WeightsOp *op) {
  std::optional<mlir::RankedTensorType> bias2Type;
  if (op->getBias_2()) {
    bias2Type = op->getBias_2().getType();
  }
  std::optional<MemoryConfigAttr> outputMemoryConfig;
  if (op->getOutputMemoryConfigAttr()) {
    outputMemoryConfig = op->getOutputMemoryConfigAttr();
  }
  return getPreparedMoEComputeW2WeightsOutputType(
      op->getOperation(), op->getW2().getType(), bias2Type, op->getHiddenSize(),
      op->getIntermediateSize(), outputMemoryConfig);
}

} // namespace mlir::tt::ttnn::op_model
