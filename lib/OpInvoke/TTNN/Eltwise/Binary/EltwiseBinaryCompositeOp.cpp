// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Eltwise/Binary/EltwiseBinaryCompositeOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

#include <optional>

namespace ttnn_op_invoke {

EltwiseBinaryCompositeResolvedParams resolveEltwiseBinaryCompositeParams(
    const ::tt::target::ttnn::EltwiseBinaryCompositeOpT
        &eltwiseBinaryCompositeOpT) {

  EltwiseBinaryCompositeResolvedParams params;

  if (eltwiseBinaryCompositeOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(
            *eltwiseBinaryCompositeOpT.out));
    LOG_ASSERT(
        operations::utils::inSystemMemory(*eltwiseBinaryCompositeOpT.out) ||
            params.outputMemoryConfig.has_value(),
        "Memory config must exist for device tensors");
  }

  return params;
}

EltwiseBinaryCompositeScalarResolvedParams
resolveEltwiseBinaryCompositeScalarParams(
    const ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT
        &eltwiseBinaryCompositeScalarOpT) {

  EltwiseBinaryCompositeScalarResolvedParams params;

  LOG_ASSERT(eltwiseBinaryCompositeScalarOpT.rhs.type ==
                     ::tt::target::ttnn::NumberType::FP ||
                 eltwiseBinaryCompositeScalarOpT.rhs.type ==
                     ::tt::target::ttnn::NumberType::I32,
             "Exponent must be either FP or I32");

  if (eltwiseBinaryCompositeScalarOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(
            *eltwiseBinaryCompositeScalarOpT.out));

    LOG_ASSERT(operations::utils::inSystemMemory(
                   *eltwiseBinaryCompositeScalarOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag, typename Scalar>
auto createEltwiseBinaryCompositeScalarTuple(
    Tag tag, TensorArg lhs, Scalar rhs,
    const EltwiseBinaryCompositeScalarResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(lhs, tag), rhs,
                         params.outputMemoryConfig);
}

EltwiseBinaryCompositeScalarOpResult callEltwiseBinaryCompositeScalar(
    CallType callType,
    const ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT
        &eltwiseBinaryCompositeScalarOpT,
    TensorArg lhs, ::ttnn::MeshDevice *device) {

  EltwiseBinaryCompositeScalarResolvedParams params =
      resolveEltwiseBinaryCompositeScalarParams(
          eltwiseBinaryCompositeScalarOpT);

  auto invokeWithScalar =
      [&](auto scalar) -> EltwiseBinaryCompositeScalarOpResult {
    auto makeTuple = [&](auto tag) {
      return createEltwiseBinaryCompositeScalarTuple(tag, lhs, scalar, params);
    };

    return callOp<EltwiseBinaryCompositeScalarOpResult>(WRAP_OP(::ttnn::pow),
                                                        callType, makeTuple,
                                                        device);
  };

  const auto &rhs = eltwiseBinaryCompositeScalarOpT.rhs;
  if (rhs.type == ::tt::target::ttnn::NumberType::FP) {
    return invokeWithScalar(rhs.AsFP()->value);
  }
  return invokeWithScalar(rhs.AsI32()->value);
}

} // namespace ttnn_op_invoke
