// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device --d2m-linalg-to-affine --d2m-insert-dst-register-access-unscheduled --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// A bf16 tile_mul is an SFPU op with BOTH operands staged in DST
// (mola-patches 34, 2026-09-05): on Blackhole the FPU's ELWMUL bf16 x bf16
// product differs from round-to-nearest-even on ~28% of random operands even
// at HiFi4, while mul_binary_tile is exact and is what TTNN uses. A bf16
// tile_add stays on the FPU (operands read from L1, no DST staging), whose
// result matches TTNN exactly.

#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

// CHECK-LABEL: func.func @bf16_mul_is_sfpu
func.func @bf16_mul_is_sfpu(
    %in0: memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>,
    %in1: memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>,
    %out0: memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [],
               iterator_types = [], threads = [#d2m.thread<unified>]}
      ins(%in0, %in1 : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>,
                       memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>)
      outs(%out0 : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1_>) {
  ^unified0:
    %arg0_cb = d2m.get_cb(0) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1_>>
    %arg1_cb = d2m.get_cb(1) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1_>>
    %arg2_cb = d2m.get_cb(2) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1_>>
    %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1_>>
        -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1_>
    %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1_>>
        -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1_>
    %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1_>>
        -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1_>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c1 step %c1 {
      scf.for %j = %c0 to %c1 step %c1 {
        %sv0 = memref.subview %cb0[%i, %j] [1, 1] [1, 1]
            : memref<1x1x!ttcore.tile<32x32, bf16>, #l1_>
            to memref<1x1x!ttcore.tile<32x32, bf16>, strided<[1, 1], offset: ?>, #l1_>
        %sv1 = memref.subview %cb1[%i, %j] [1, 1] [1, 1]
            : memref<1x1x!ttcore.tile<32x32, bf16>, #l1_>
            to memref<1x1x!ttcore.tile<32x32, bf16>, strided<[1, 1], offset: ?>, #l1_>
        %sv2 = memref.subview %cb2[%i, %j] [1, 1] [1, 1]
            : memref<1x1x!ttcore.tile<32x32, bf16>, #l1_>
            to memref<1x1x!ttcore.tile<32x32, bf16>, strided<[1, 1], offset: ?>, #l1_>
        linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                             affine_map<(d0, d1) -> (d0, d1)>,
                             affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]}
            ins(%sv0, %sv1 : memref<1x1x!ttcore.tile<32x32, bf16>, strided<[1, 1], offset: ?>, #l1_>,
                             memref<1x1x!ttcore.tile<32x32, bf16>, strided<[1, 1], offset: ?>, #l1_>)
            outs(%sv2 : memref<1x1x!ttcore.tile<32x32, bf16>, strided<[1, 1], offset: ?>, #l1_>) {
        ^bb0(%a: !ttcore.tile<32x32, bf16>, %b: !ttcore.tile<32x32, bf16>,
             %c: !ttcore.tile<32x32, bf16>):
          %mul = "d2m.tile_mul"(%a, %b)
              : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
          linalg.yield %mul : !ttcore.tile<32x32, bf16>
        }
      }
    }
  }
  return
}
// Both operands are staged into DST before the multiply reads them.
// CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<{{[0-9]+}}x!ttcore.tile<32x32, bf16>, #dst>
// CHECK: affine.store {{.*}}, %[[DST]][{{.*}}]
// CHECK: affine.store {{.*}}, %[[DST]][{{.*}} + 1]
// CHECK: %[[IN0_DST:.*]] = affine.load %[[DST]][{{.*}}]
// CHECK: %[[IN1_DST:.*]] = affine.load %[[DST]][{{.*}} + 1]
// CHECK: %[[MUL:.*]] = "d2m.tile_mul"(%[[IN0_DST]], %[[IN1_DST]])
// CHECK: affine.store %[[MUL]], %[[DST]][{{.*}}]
