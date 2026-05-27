// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_TTNNOUTPUTTENSORINFERENCE_H
#define TTMLIR_OPMODEL_TTNN_TTNNOUTPUTTENSORINFERENCE_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::op_model {

// TODO(jserbedzija): This logic should be moved to tt-metal side.
// Metal issue: https://github.com/tenstorrent/tt-metal/issues/21061

// Calculate the output tensor type of the prepared weights for a conv2d op.
// Conv2dConfigAttr is used to determine the output tensor type.
mlir::RankedTensorType
getPreparedConv2dWeightsOutputTensor(Conv2dOp *op,
                                     Conv2dConfigAttr conv2dConfig);

// Calculate the output tensor type of the prepared weights for a
// conv_transpose2d op. Conv2dConfigAttr is used to determine the output tensor
// type.
mlir::RankedTensorType
getPreparedConvTranspose2dWeightsOutputTensor(ConvTranspose2dOp *op,
                                              Conv2dConfigAttr conv2dConfig);

// Calculate the output tensor type for prepare_moe_compute_w0_w1_weights.
// Queries the shared TTNN-op chain via graph capture to derive the
// 6-D packed shape, the DRAM HEIGHT_SHARDED MemoryConfig (with the
// bank-permuted CoreRangeSet), and the dtype. Swap to query_op_constraints
// against ttnn::experimental::prepare_moe_compute_w0_w1 once tt-metal lands
// its C++ entry.
mlir::RankedTensorType
getPreparedMoEComputeW0W1WeightsOutputType(PrepareMoEComputeW0W1WeightsOp *op);

// Op-free version for use during TTIR -> TTNN conversion, when the TTNN
// PrepareMoEComputeW0W1WeightsOp does not yet exist. `anchor` is any op in
// the module being lowered, used only for device-grid lookup.
mlir::RankedTensorType getPreparedMoEComputeW0W1WeightsOutputType(
    mlir::Operation *anchor, mlir::RankedTensorType w0Type,
    mlir::RankedTensorType w1Type,
    std::optional<mlir::RankedTensorType> bias0Type,
    std::optional<mlir::RankedTensorType> bias1Type, uint32_t hiddenSize,
    uint32_t intermediateSize,
    std::optional<MemoryConfigAttr> outputMemoryConfig);

// Same as above but for the W2 path.
mlir::RankedTensorType
getPreparedMoEComputeW2WeightsOutputType(PrepareMoEComputeW2WeightsOp *op);

// Op-free version for use during TTIR -> TTNN conversion.
mlir::RankedTensorType getPreparedMoEComputeW2WeightsOutputType(
    mlir::Operation *anchor, mlir::RankedTensorType w2Type,
    std::optional<mlir::RankedTensorType> bias2Type, uint32_t hiddenSize,
    uint32_t intermediateSize,
    std::optional<MemoryConfigAttr> outputMemoryConfig);

} // namespace mlir::tt::ttnn::op_model

#endif // TTMLIR_OPMODEL_TTNN_TTNNOUTPUTTENSORINFERENCE_H
