// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// An integer-keyed sort whose indices are used is decomposed by
// SortOpRewritePattern into a rank-by-comparison permutation computed in f32
// (ttnn.sort's bf16 key comparison returns wrong indices for integer / >256
// keys -- tt-metal #46331). The sorted values still come from ttnn.sort.

module {
  func.func public @argsort_integer(%arg0: tensor<1x10xsi32>) -> (tensor<1x10xsi32>, tensor<1x10xsi32>) {
    // CHECK-LABEL: func.func public @argsort_integer
    // Keys are cast to f32 (not ui16) so the comparison is exact.
    // CHECK: "ttnn.typecast"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
    // Position indices for the comparison / inverse permutation.
    // CHECK: "ttnn.arange"
    // rank-by-comparison: compares + reduce to a rank, then one-hot + reduce.
    // CHECK: "ttnn.lt"
    // CHECK: "ttnn.eq"
    // CHECK: "ttnn.sum"
    // CHECK: "ttnn.eq"
    // CHECK: "ttnn.sum"
    // Indices cast back to the integer index dtype.
    // CHECK: "ttnn.typecast"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>
    // Values still come from ttnn.sort.
    // CHECK: "ttnn.sort"
    %values, %indices = "ttir.sort"(%arg0) <{descending = false, dim = 1 : si32, stable = false}> : (tensor<1x10xsi32>) -> (tensor<1x10xsi32>, tensor<1x10xsi32>)
    return %values, %indices : tensor<1x10xsi32>, tensor<1x10xsi32>
  }

  // A value-only integer sort (indices unused) must NOT be decomposed -- the
  // ttnn.sort values are correct, so it keeps the plain lowering.
  func.func public @sort_values_only(%arg0: tensor<1x10xsi32>) -> tensor<1x10xsi32> {
    // CHECK-LABEL: func.func public @sort_values_only
    // CHECK-NOT: "ttnn.arange"
    // CHECK: "ttnn.sort"
    %values, %indices = "ttir.sort"(%arg0) <{descending = false, dim = 1 : si32, stable = false}> : (tensor<1x10xsi32>) -> (tensor<1x10xsi32>, tensor<1x10xsi32>)
    return %values : tensor<1x10xsi32>
  }
}
