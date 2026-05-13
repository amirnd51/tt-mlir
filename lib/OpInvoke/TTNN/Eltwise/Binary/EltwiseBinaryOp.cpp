// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Eltwise/Binary/EltwiseBinaryOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

#include <optional>

namespace ttnn_op_invoke {

EltwiseBinaryResolvedParams resolveEltwiseBinaryParams(
    const ::tt::target::ttnn::EltwiseBinaryOpT &eltwiseBinaryOpT) {

  EltwiseBinaryResolvedParams params;

  if (eltwiseBinaryOpT.out) {
    params.outputDType = operations::utils::getDataType(*eltwiseBinaryOpT.out);
  } else if (eltwiseBinaryOpT.output_dtype.has_value()) {
    params.outputDType = operations::utils::toTTNNDataType(
        eltwiseBinaryOpT.output_dtype.value());
  }

  if (eltwiseBinaryOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*eltwiseBinaryOpT.out));
    LOG_ASSERT(operations::utils::inSystemMemory(*eltwiseBinaryOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

} // namespace ttnn_op_invoke
