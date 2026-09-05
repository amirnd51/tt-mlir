// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// MOLA local patch 33: a reshape that re-partitions the innermost dimension
// into or out of an extent that is not a multiple of the tile width is
// miscompiled when its operand lives in DRAM (the view's addresses are correct
// in every inspectable layer; resolving them against a DRAM-sharded buffer
// truncates each row start to a 32-element granule). The same view over an L1
// operand is exact. The operand of such a reshape is therefore laid out in L1
// by its own to_layout and the view is taken over the L1 buffer; aligned
// reshapes keep the role default (DRAM inputs here).

// The unaligned split's operand is laid out in L1 ...
// CHECK-DAG: #ttcore.metal_layout<logical_shape = 1x128x640, {{.*}}, l1, {{.*}}>
// ... and the aligned split's operand stays in DRAM.
// CHECK-DAG: #ttcore.metal_layout<logical_shape = 1x64x640, {{.*}}, dram, {{.*}}>

module {
  // CHECK-LABEL: func.func @split_unaligned
  func.func @split_unaligned(%arg0: tensor<1x128x640xbf16>) -> tensor<1x128x16x40xbf16> {
    // CHECK: d2m.view_layout
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 128 : i32, 16 : i32, 40 : i32]}> : (tensor<1x128x640xbf16>) -> tensor<1x128x16x40xbf16>
    return %0 : tensor<1x128x16x40xbf16>
  }

  // CHECK-LABEL: func.func @split_aligned
  func.func @split_aligned(%arg0: tensor<1x64x640xbf16>) -> tensor<1x64x20x32xbf16> {
    // CHECK: d2m.view_layout
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 64 : i32, 20 : i32, 32 : i32]}> : (tensor<1x64x640xbf16>) -> tensor<1x64x20x32xbf16>
    return %0 : tensor<1x64x20x32xbf16>
  }
}
