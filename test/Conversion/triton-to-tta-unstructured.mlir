// RUN: triton-xyz-opt --split-input-file --triton-to-unstructured --triton-to-tta-unstructured --remove-dead-values --canonicalize %s | FileCheck %s

module {
// CHECK-LABEL: tt.func public @masked_gather_scatter(
// CHECK: %[[OFF:.*]] = tt.make_range
// CHECK: %[[MASK:.*]] = arith.cmpi slt, %[[OFF]]
// CHECK: %[[A0:.*]] = tta.make_addr %arg0 to sizes: [4]
// CHECK: %[[R0:.*]] = "tta.reindex"(%[[A0]], %[[OFF]], %[[MASK]])
// CHECK: %[[V:.*]] = "tta.load"(%[[R0]], %{{.*}})
// CHECK: %[[A1:.*]] = tta.make_addr %arg1 to sizes: [4]
// CHECK: %[[R1:.*]] = "tta.reindex"(%[[A1]], %[[OFF]], %[[MASK]])
// CHECK: "tta.store"(%[[R1]], %[[V]])
  tt.func public @masked_gather_scatter(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %limit = tt.splat %arg2 : i32 -> tensor<4xi32>
    %mask = arith.cmpi slt, %range, %limit : tensor<4xi32>
    %zero = arith.constant 0.0 : f32
    %other = tt.splat %zero : f32 -> tensor<4xf32>
    %val = tt.load %in_ptrs, %mask, %other : tensor<4x!tt.ptr<f32>>
    tt.store %out_ptrs, %val, %mask : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func public @offset_width_upgrade(
// CHECK: %[[OFF64:.*]] = arith.addi
// CHECK: %[[A0:.*]] = tta.make_addr %arg0 to sizes: [4]
// CHECK: %[[R0:.*]] = "tta.reindex"(%[[A0]], %[[OFF64]])
// CHECK: %[[A1:.*]] = tta.make_addr %arg1 to sizes: [4]
// CHECK: %[[R1:.*]] = "tta.reindex"(%[[A1]], %{{.*}})
// CHECK: "tta.store"(%[[R1]], %{{.*}})
  tt.func public @offset_width_upgrade(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %c5_i64 = arith.constant 5 : i64
    %off64 = tt.splat %c5_i64 : i64 -> tensor<4xi64>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptr0 = tt.addptr %base, %off64 : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
    %ptr1 = tt.addptr %ptr0, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %val = tt.load %ptr1 : tensor<4x!tt.ptr<f32>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %out_ptrs, %val : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func public @loop_ptr_iter_args(
// CHECK: scf.for
// CHECK: %[[A0:.*]] = tta.make_addr %arg0 to sizes: [4]
// CHECK: %[[R0:.*]] = "tta.reindex"(%[[A0]], %{{.*}})
// CHECK: %[[V:.*]] = "tta.load"(%[[R0]], %{{.*}})
// CHECK: %[[A1:.*]] = tta.make_addr %arg1 to sizes: [4]
// CHECK: %[[R1:.*]] = "tta.reindex"(%[[A1]], %{{.*}})
// CHECK: "tta.store"(%[[R1]], %[[V]])
  tt.func public @loop_ptr_iter_args(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %out_ptrs = tt.addptr %out_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %res:2 = scf.for %iv = %c0_i32 to %arg2 step %c1_i32 iter_args(%in = %in_ptrs, %out = %out_ptrs) -> (tensor<4x!tt.ptr<f32>>, tensor<4x!tt.ptr<f32>>)  : i32 {
      %val = tt.load %in : tensor<4x!tt.ptr<f32>>
      tt.store %out, %val : tensor<4x!tt.ptr<f32>>
      %next_in = tt.addptr %in, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %next_out = tt.addptr %out, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %next_in, %next_out : tensor<4x!tt.ptr<f32>>, tensor<4x!tt.ptr<f32>>
    }
    tt.return
  }
}
