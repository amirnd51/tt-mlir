// RUN: ttmlir-opt --d2m-materialize-reduction-scalers --d2m-mark-synchronized-buffers %s | FileCheck %s

#l1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>

module {
  // CHECK-LABEL: func.func @materialize_tile_fill_scaler
  func.func @materialize_tile_fill_scaler() {
    %outer = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins()
        outs(%outer : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
    ^unified0:
      %in = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %out = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>

      // CHECK: %[[SCALER:.*]] = memref.alloc() {{.*}}d2m.materialized_reduction_scaler{{.*}}d2m.synchronized_buffer = 1 : i32
      // CHECK: linalg.generic
      // CHECK-SAME: outs(%[[SCALER]]
      // CHECK: d2m.tile_fill
      // CHECK: linalg.generic
      // CHECK-SAME: ins(%{{.*}}, %[[SCALER]]
      // CHECK: ^bb0(%[[IN_TILE:.*]]: !ttcore.tile<32x32, f32>, %[[SCALER_TILE:.*]]: !ttcore.tile<32x32, f32>, %[[OUT_TILE:.*]]: !ttcore.tile<32x32, f32>):
      // CHECK: d2m.tile_reduce_sum
      // CHECK-SAME: %[[IN_TILE]], %[[SCALER_TILE]]
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
          ins(%in : memref<1x1x!ttcore.tile<32x32, f32>, #l1>)
          outs(%out : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%a: !ttcore.tile<32x32, f32>, %c: !ttcore.tile<32x32, f32>):
        %one = arith.constant 1.000000e+00 : f32
        %scale = d2m.tile_fill(%one) : f32 -> !ttcore.tile<32x32, f32>
        %zero = arith.constant 0.000000e+00 : f32
        %acc = d2m.tile_fill(%zero) : f32 -> !ttcore.tile<32x32, f32>
        %r = "d2m.tile_reduce_sum"(%a, %scale, %acc) <{reduce_dim = #d2m<reduce_dim R>}> : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %r : !ttcore.tile<32x32, f32>
      }
    }
    return
  }
}
