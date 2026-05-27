// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_SHAREDTTNN_MOECOMPUTEWEIGHTPREP_H
#define TTMLIR_SHAREDTTNN_MOECOMPUTEWEIGHTPREP_H

// Thin adapters over tt-metal's C++ moe_compute weight-prep helpers
// (`ttnn::experimental::prepare_w0_w1_tensor_for_moe_compute` and friends, see
// ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/moe_compute_utils.hpp).
//
// The actual packing logic lives entirely in tt-metal now; this header only
// wires the runtime / OpModel call sites to it:
//   - convert weights/biases to ROW_MAJOR (the tt-metal packers reshape the raw
//     layout and only stay semantics-preserving on row-major data — the
//     parity tests upload their inputs as ROW_MAJOR for the same reason),
//   - dispatch to the bias / no-bias packer,
//   - pick the DRAM HEIGHT_SHARDED memory config (caller override, else the
//     bank-permuted layout from `get_weight_mem_configs`),
//   - host round-trip the packed tensor into that config via
//     `quantize_weights_via_host` (avoids the interleaved_to_sharded kernel
//     landing on DRAM-bank workers outside the compute grid).

#include <cstdint>
#include <optional>

#include "tt-metalium/mesh_device.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/experimental/ccl/moe_compute/moe_compute_utils.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace mlir::tt::ttnn::shared {

namespace detail {

// A caller-supplied config other than the default INTERLEAVED/DRAM means the
// caller wants something specific; respect it. Otherwise fall back to the
// bank-permuted HEIGHT_SHARDED DRAM layout tt-metal computes.
inline bool isCallerOverride(const ::tt::tt_metal::MemoryConfig &cfg) {
  return cfg.memory_layout() !=
             ::tt::tt_metal::TensorMemoryLayout::INTERLEAVED ||
         cfg.buffer_type() != ::tt::tt_metal::BufferType::DRAM;
}

inline ::ttnn::Tensor toRowMajor(const ::ttnn::Tensor &t) {
  return ::ttnn::to_layout(t, ::ttnn::Layout::ROW_MAJOR, std::nullopt,
                           std::nullopt);
}

} // namespace detail

// Pack W0/W1 (optionally with biases) into the layout the moe_compute kernel
// reads and place it under the DRAM HEIGHT_SHARDED config.
inline ::ttnn::Tensor prepare_moe_compute_w0_w1(
    const ::ttnn::Tensor &w0, const ::ttnn::Tensor &w1,
    std::optional<::ttnn::Tensor> b0, std::optional<::ttnn::Tensor> b1,
    uint32_t hidden_size, uint32_t intermediate_size,
    ::ttnn::MeshDevice *device,
    const ::tt::tt_metal::MemoryConfig &output_memory_config) {
  const auto &w0_shape = w0.logical_shape();
  uint32_t L = w0_shape[0];
  uint32_t E = w0_shape[1];
  bool has_bias = b0.has_value();

  ::ttnn::Tensor w0_rm = detail::toRowMajor(w0);
  ::ttnn::Tensor w1_rm = detail::toRowMajor(w1);

  ::ttnn::Tensor packed;
  if (has_bias) {
    auto b0_rm = detail::toRowMajor(
        ::ttnn::reshape(*b0, ::ttnn::Shape({L, E, intermediate_size})));
    auto b1_rm = detail::toRowMajor(
        ::ttnn::reshape(*b1, ::ttnn::Shape({L, E, intermediate_size})));
    packed = ::ttnn::experimental::prepare_w0_w1_tensor_with_bias(
        w0_rm, w1_rm, b0_rm, b1_rm, L, E, hidden_size, intermediate_size);
  } else {
    packed = ::ttnn::experimental::prepare_w0_w1_tensor_for_moe_compute(
        w0_rm, w1_rm, L, E, hidden_size, intermediate_size);
  }

  ::tt::tt_metal::MemoryConfig final_mem_config =
      detail::isCallerOverride(output_memory_config)
          ? output_memory_config
          : ::ttnn::experimental::get_weight_mem_configs(
                device, L, E, hidden_size, intermediate_size, has_bias)
                .w0_w1;
  return ::ttnn::experimental::quantize_weights_via_host(
      packed, packed.dtype(), final_mem_config);
}

// Pack W2 (optionally with bias) into the layout the moe_compute kernel reads
// and place it under the DRAM HEIGHT_SHARDED config.
inline ::ttnn::Tensor prepare_moe_compute_w2(
    const ::ttnn::Tensor &w2, std::optional<::ttnn::Tensor> b2,
    uint32_t hidden_size, uint32_t intermediate_size,
    ::ttnn::MeshDevice *device,
    const ::tt::tt_metal::MemoryConfig &output_memory_config) {
  const auto &w2_shape = w2.logical_shape();
  uint32_t L = w2_shape[0];
  uint32_t E = w2_shape[1];
  bool has_bias = b2.has_value();

  ::ttnn::Tensor w2_rm = detail::toRowMajor(w2);

  ::ttnn::Tensor packed;
  if (has_bias) {
    auto b2_rm = detail::toRowMajor(
        ::ttnn::reshape(*b2, ::ttnn::Shape({L, E, hidden_size})));
    packed = ::ttnn::experimental::prepare_w2_tensor_with_bias(
        w2_rm, b2_rm, L, E, intermediate_size, hidden_size);
  } else {
    packed = ::ttnn::experimental::prepare_w2_tensor_for_moe_compute(
        w2_rm, L, E, intermediate_size, hidden_size);
  }

  ::tt::tt_metal::MemoryConfig final_mem_config =
      detail::isCallerOverride(output_memory_config)
          ? output_memory_config
          : ::ttnn::experimental::get_weight_mem_configs(
                device, L, E, hidden_size, intermediate_size, has_bias)
                .w2;
  return ::ttnn::experimental::quantize_weights_via_host(
      packed, packed.dtype(), final_mem_config);
}

} // namespace mlir::tt::ttnn::shared

#endif // TTMLIR_SHAREDTTNN_MOECOMPUTEWEIGHTPREP_H
