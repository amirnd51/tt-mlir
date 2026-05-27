// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/prepare_moe_compute_w0_w1_weights.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/SharedTTNN/MoEComputeWeightPrep.h"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::PrepareMoEComputeW0W1WeightsOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &w0 = tensorPool.getTTNNTensorAndValidate(op->w0());
  const ::ttnn::Tensor &w1 = tensorPool.getTTNNTensorAndValidate(op->w1());

  std::optional<::ttnn::Tensor> bias0;
  if (op->bias_0() != nullptr) {
    bias0 = tensorPool.getTTNNTensorAndValidate(op->bias_0());
  }
  std::optional<::ttnn::Tensor> bias1;
  if (op->bias_1() != nullptr) {
    bias1 = tensorPool.getTTNNTensorAndValidate(op->bias_1());
  }

  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();
  ::ttnn::MemoryConfig outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->output_memory_config())
          .value_or(::ttnn::DRAM_MEMORY_CONFIG);

  ::ttnn::Tensor out = mlir::tt::ttnn::shared::prepare_moe_compute_w0_w1(
      w0, w1, bias0, bias1, op->hidden_size(), op->intermediate_size(),
      &meshDevice, outputMemoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
