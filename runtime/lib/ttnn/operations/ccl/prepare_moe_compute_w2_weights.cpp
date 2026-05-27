// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/prepare_moe_compute_w2_weights.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/SharedTTNN/MoEComputeWeightPrep.h"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::PrepareMoEComputeW2WeightsOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &w2 = tensorPool.getTTNNTensorAndValidate(op->w2());

  std::optional<::ttnn::Tensor> bias2;
  if (op->bias_2() != nullptr) {
    bias2 = tensorPool.getTTNNTensorAndValidate(op->bias_2());
  }

  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();
  ::ttnn::MemoryConfig outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->output_memory_config())
          .value_or(::ttnn::DRAM_MEMORY_CONFIG);

  ::ttnn::Tensor out = mlir::tt::ttnn::shared::prepare_moe_compute_w2(
      w2, bias2, op->hidden_size(), op->intermediate_size(), &meshDevice,
      outputMemoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
