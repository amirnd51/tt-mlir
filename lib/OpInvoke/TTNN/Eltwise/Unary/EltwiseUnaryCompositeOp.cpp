// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Eltwise/Unary/EltwiseUnaryCompositeOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <tuple>
#include <variant>

namespace ttnn_op_invoke {

EltwiseUnaryCompositeResolvedParams resolveEltwiseUnaryCompositeParams(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT) {

  EltwiseUnaryCompositeResolvedParams params;

  params.fastApproxMode = false;

  if (eltwiseUnaryCompositeOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(
            *eltwiseUnaryCompositeOpT.out));
    LOG_ASSERT(
        operations::utils::inSystemMemory(*eltwiseUnaryCompositeOpT.out) ||
            params.outputMemoryConfig.has_value(),
        "Memory config must exist for device tensors");
  }

  return params;
}

EltwiseUnaryCompositeClampScalarResolvedParams
resolveEltwiseUnaryCompositeClampScalarParams(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT) {

  EltwiseUnaryCompositeClampScalarResolvedParams params;

  const auto *clampParams =
      eltwiseUnaryCompositeOpT.params.AsClampScalarOpParams();

  LOG_ASSERT(clampParams->min.type == clampParams->max.type,
             "Clamp scalar min/max types must match");

  switch (clampParams->min.type) {
  case ::tt::target::ttnn::NumberType::FP:
    params.min = clampParams->min.AsFP()->value;
    params.max = clampParams->max.AsFP()->value;
    break;
  case ::tt::target::ttnn::NumberType::I32:
    params.min = clampParams->min.AsI32()->value;
    params.max = clampParams->max.AsI32()->value;
    break;
  default:
    LOG_FATAL("unknown clamp scalar min/max type");
  }

  if (eltwiseUnaryCompositeOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(
            *eltwiseUnaryCompositeOpT.out));
    LOG_ASSERT(
        operations::utils::inSystemMemory(*eltwiseUnaryCompositeOpT.out) ||
            params.outputMemoryConfig.has_value(),
        "Memory config must exist for device tensors");
  }

  return params;
}

EltwiseUnaryCompositeClampTensorResolvedParams
resolveEltwiseUnaryCompositeClampTensorParams(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT) {

  EltwiseUnaryCompositeClampTensorResolvedParams params;

  if (eltwiseUnaryCompositeOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(
            *eltwiseUnaryCompositeOpT.out));
    LOG_ASSERT(
        operations::utils::inSystemMemory(*eltwiseUnaryCompositeOpT.out) ||
            params.outputMemoryConfig.has_value(),
        "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createEltwiseUnaryCompositeClampScalarTuple(
    Tag tag,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    TensorArg input,
    const EltwiseUnaryCompositeClampScalarResolvedParams &params) {
  LOG_ASSERT(params.min.has_value() && params.max.has_value(),
             "clamp scalar min/max not resolved");
  return std::make_tuple(resolveTensorArg(input, tag), *params.min, *params.max,
                         params.outputMemoryConfig);
}

EltwiseUnaryCompositeOpResult callEltwiseUnaryCompositeClampScalar(
    CallType callType,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    TensorArg input, ::ttnn::MeshDevice *device) {

  EltwiseUnaryCompositeClampScalarResolvedParams params =
      resolveEltwiseUnaryCompositeClampScalarParams(eltwiseUnaryCompositeOpT);

  auto makeTuple = [&](auto tag) {
    return createEltwiseUnaryCompositeClampScalarTuple(
        tag, eltwiseUnaryCompositeOpT, input, params);
  };

  callOp(::ttnn::clamp);
}

template <typename Tag>
auto createEltwiseUnaryCompositeClampTensorTuple(
    Tag tag,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    TensorArg input, TensorArg min, TensorArg max,
    const EltwiseUnaryCompositeClampTensorResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag),
                         resolveTensorArg(min, tag), resolveTensorArg(max, tag),
                         params.outputMemoryConfig);
}

EltwiseUnaryCompositeOpResult callEltwiseUnaryCompositeClampTensor(
    CallType callType,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    TensorArg input, TensorArg min, TensorArg max, ::ttnn::MeshDevice *device) {

  EltwiseUnaryCompositeClampTensorResolvedParams params =
      resolveEltwiseUnaryCompositeClampTensorParams(eltwiseUnaryCompositeOpT);

  auto makeTuple = [&](auto tag) {
    return createEltwiseUnaryCompositeClampTensorTuple(
        tag, eltwiseUnaryCompositeOpT, input, min, max, params);
  };

  callOp(::ttnn::clamp);
}

} // namespace ttnn_op_invoke
