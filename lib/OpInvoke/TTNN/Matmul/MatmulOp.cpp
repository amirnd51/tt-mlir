// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Matmul/MatmulOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttmlir/Target/TTNN/operations/matmul_generated.h"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

// Returns true if the matmul program config already carries a fused
// activation.
static bool programCarriesFusedActivation(
    const std::optional<::ttnn::operations::matmul::MatmulProgramConfig> &pc) {
  if (!pc) {
    return false;
  }
  return std::visit(
      [](const auto &cfg) -> bool {
        using T = std::decay_t<decltype(cfg)>;
        if constexpr (
            std::is_same_v<T, ::ttnn::operations::matmul::
                                  MatmulMultiCoreReuseMultiCastProgramConfig> ||
            std::is_same_v<T,
                           ::ttnn::operations::matmul::
                               MatmulMultiCoreReuseMultiCast1DProgramConfig> ||
            std::is_same_v<
                T, ::ttnn::operations::matmul::
                       MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig> ||
            std::is_same_v<
                T,
                ::ttnn::operations::matmul::
                    MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig>) {
          return cfg.fused_activation.has_value();
        }
        return false;
      },
      *pc);
}

MatmulResolvedParams
resolveMatmulParams(const ::tt::target::ttnn::MatmulOpT &matmulOpT) {

  MatmulResolvedParams params;

  if (matmulOpT.out) {
    params.outputDType = operations::utils::getDataType(*matmulOpT.out);
  }
  params.matmulProgramConfig =
      operations::utils::createMatmulProgramConfigIfNeeded(matmulOpT);
  if (!matmulOpT.activation.empty() &&
      !programCarriesFusedActivation(params.matmulProgramConfig)) {
    params.activation = std::make_optional(matmulOpT.activation);
  }
  if (matmulOpT.compute_config) {
    params.computeConfig = operations::utils::createDeviceComputeKernelConfig(
        *matmulOpT.compute_config);
  }

  if (matmulOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*matmulOpT.out));
    LOG_ASSERT(operations::utils::inSystemMemory(*matmulOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createMatmulTuple(Tag tag, const ::tt::target::ttnn::MatmulOpT &matmulOpT,
                       TensorArg lhs, TensorArg rhs,
                       const MatmulResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(lhs, tag), resolveTensorArg(rhs, tag),
      matmulOpT.transpose_a, matmulOpT.transpose_b, params.outputMemoryConfig,
      params.outputDType, params.matmulProgramConfig, params.activation,
      params.computeConfig, /*core_grid=*/std::nullopt,
      /*output_tile=*/std::nullopt, /*optional_output_tensor=*/std::nullopt,
      /*global_cb=*/std::nullopt, /*sub_device_id=*/std::nullopt);
}

MatmulOpResult callMatmul(CallType callType,
                          const ::tt::target::ttnn::MatmulOpT &matmulOpT,
                          TensorArg lhs, TensorArg rhs,
                          ::ttnn::MeshDevice *device) {

  MatmulResolvedParams params = resolveMatmulParams(matmulOpT);

  auto makeTuple = [&](auto tag) {
    return createMatmulTuple(tag, matmulOpT, lhs, rhs, params);
  };

  return callOp<MatmulOpResult>(WRAP_OP(::ttnn::matmul), callType, makeTuple,
                                device);
}

LinearResolvedParams
resolveLinearParams(const ::tt::target::ttnn::LinearOpT &linearOpT) {

  LinearResolvedParams params;

  if (linearOpT.out) {
    params.outputDType = operations::utils::getDataType(*linearOpT.out);
  }
  params.matmulProgramConfig =
      operations::utils::createMatmulProgramConfigIfNeeded(linearOpT);
  if (!linearOpT.activation.empty() &&
      !programCarriesFusedActivation(params.matmulProgramConfig)) {
    params.activation = std::make_optional(linearOpT.activation);
  }
  if (linearOpT.compute_config) {
    params.computeConfig = operations::utils::createDeviceComputeKernelConfig(
        *linearOpT.compute_config);
  }

  if (linearOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*linearOpT.out));
    LOG_ASSERT(operations::utils::inSystemMemory(*linearOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createLinearTuple(Tag tag, const ::tt::target::ttnn::LinearOpT &linearOpT,
                       TensorArg a, TensorArg b,
                       const std::optional<TensorArg> bias,
                       const LinearResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(a, tag), resolveTensorArg(b, tag),
      bias.has_value()
          ? std::make_optional(*std::get<const ::ttnn::Tensor *>(bias.value()))
          : std::nullopt,
      linearOpT.transpose_a, linearOpT.transpose_b, params.outputMemoryConfig,
      params.outputDType, params.matmulProgramConfig, params.activation,
      params.computeConfig,
      /*core_grid=*/std::nullopt, /*output_tile=*/std::nullopt,
      /*optional_output_tensor=*/std::nullopt,
      /*global_cb=*/std::nullopt, /*sub_device_id=*/std::nullopt);
}

LinearOpResult callLinear(CallType callType,
                          const ::tt::target::ttnn::LinearOpT &linearOpT,
                          TensorArg a, TensorArg b,
                          const std::optional<TensorArg> bias,
                          ::ttnn::MeshDevice *device) {
  LinearResolvedParams params = resolveLinearParams(linearOpT);

  auto makeTuple = [&](auto tag) {
    return createLinearTuple(tag, linearOpT, a, b, bias, params);
  };

  return callOp<LinearOpResult>(WRAP_OP(::ttnn::linear), callType, makeTuple,
                                device);
}

SparseMatmulResolvedParams resolveSparseMatmulParams(
    const ::tt::target::ttnn::SparseMatmulOpT &sparseMatmulOpT) {

  SparseMatmulResolvedParams params;

  if (sparseMatmulOpT.nnz) {
    params.nnz =
        std::make_optional(static_cast<uint32_t>(*sparseMatmulOpT.nnz));
  }

  auto matmulProgramConfig =
      operations::utils::createMatmulProgramConfigIfNeeded(sparseMatmulOpT);
  LOG_ASSERT(matmulProgramConfig.has_value(),
             "SparseMatmulOp requires program_config to be set at compile "
             "time");
  params.matmulProgramConfig = matmulProgramConfig.value();

  if (sparseMatmulOpT.compute_config) {
    params.computeConfig = operations::utils::createDeviceComputeKernelConfig(
        *sparseMatmulOpT.compute_config);
  }

  if (sparseMatmulOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*sparseMatmulOpT.out));
    LOG_ASSERT(operations::utils::inSystemMemory(*sparseMatmulOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createSparseMatmulTuple(
    Tag tag, const ::tt::target::ttnn::SparseMatmulOpT &sparseMatmulOpT,
    TensorArg a, TensorArg b, TensorArg sparsity,
    const SparseMatmulResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(a, tag), resolveTensorArg(b, tag),
      resolveTensorArg(sparsity, tag), params.matmulProgramConfig, params.nnz,
      sparseMatmulOpT.is_input_a_sparse, sparseMatmulOpT.is_input_b_sparse,
      params.outputMemoryConfig, /*dtype=*/std::nullopt, params.computeConfig,
      /*core_grid=*/std::nullopt, /*output_tile=*/std::nullopt);
}

SparseMatmulOpResult
callSparseMatmul(CallType callType,
                 const ::tt::target::ttnn::SparseMatmulOpT &sparseMatmulOpT,
                 TensorArg a, TensorArg b, TensorArg sparsity,
                 ::ttnn::MeshDevice *device) {
  SparseMatmulResolvedParams params =
      resolveSparseMatmulParams(sparseMatmulOpT);

  auto makeTuple = [&](auto tag) {
    return createSparseMatmulTuple(tag, sparseMatmulOpT, a, b, sparsity,
                                   params);
  };

  return callOp<SparseMatmulOpResult, false, false>(
      WRAP_OP(::ttnn::sparse_matmul), callType, makeTuple, device,
      "SparseMatmulOp");
}

} // namespace ttnn_op_invoke
