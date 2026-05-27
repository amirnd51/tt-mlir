// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline="mesh-shape=4,8" %s | FileCheck %s
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// MoE compute weight-prep ops TTIR -> TTNN lowering test. The fused
// `ttir.moe_compute` op needs the full workaround / sharding-allocation
// pipeline (which reshapes outputs 0/3/4 to height-sharded) — that lowering
// is covered by the on-device pipeline test, not this unit test.

// ttir.prepare_moe_compute_w0_w1_weights -> ttnn.prepare_moe_compute_w0_w1_weights
module attributes {} {
  func.func @prepare_w0_w1_no_bias(
      %w0: tensor<1x16x256x128xbf16>,
      %w1: tensor<1x16x256x128xbf16>)
    -> tensor<8x1x16x32x256x128xbf16> {
    %0 = "ttir.prepare_moe_compute_w0_w1_weights"(%w0, %w1)
      <{operandSegmentSizes = array<i32: 1, 1, 0, 0>,
        hidden_size = 256 : ui32,
        intermediate_size = 128 : ui32}>
      : (tensor<1x16x256x128xbf16>, tensor<1x16x256x128xbf16>)
      -> tensor<8x1x16x32x256x128xbf16>
    return %0 : tensor<8x1x16x32x256x128xbf16>
  }
}
// CHECK-LABEL: @prepare_w0_w1_no_bias
// CHECK: "ttnn.prepare_moe_compute_w0_w1_weights"
// CHECK-SAME: hidden_size = 256
// CHECK-SAME: intermediate_size = 128

// -----

// ttir.prepare_moe_compute_w2_weights -> ttnn.prepare_moe_compute_w2_weights
module attributes {} {
  func.func @prepare_w2_no_bias(
      %w2: tensor<1x16x128x256xbf16>)
    -> tensor<8x1x16x32x128x256xbf16> {
    %0 = "ttir.prepare_moe_compute_w2_weights"(%w2)
      <{hidden_size = 256 : ui32,
        intermediate_size = 128 : ui32}>
      : (tensor<1x16x128x256xbf16>) -> tensor<8x1x16x32x128x256xbf16>
    return %0 : tensor<8x1x16x32x128x256xbf16>
  }
}
// CHECK-LABEL: @prepare_w2_no_bias
// CHECK: "ttnn.prepare_moe_compute_w2_weights"
// CHECK-SAME: hidden_size = 256
// CHECK-SAME: intermediate_size = 128
