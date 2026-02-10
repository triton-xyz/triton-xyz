// RUN: triton-xyz-opt --split-input-file --triton-to-tta-structured --triton-to-tta-unstructured --remove-dead-values --canonicalize %s | FileCheck %s
// CHECK-NOT: tts.

module {
// CHECK-LABEL: tt.func public @masked_gather_scatter(
// CHECK: %[[A0:.*]] = tta.make_addr %arg0 to sizes: [4]
// CHECK: %[[V:.*]] = "tta.load"(%[[A0]], %{{.*}}, %{{.*}})
// CHECK: %[[A1:.*]] = tta.make_addr %arg1 to sizes: [4]
// CHECK: "tta.store"(%[[A1]], %[[V]], %{{.*}})
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
// CHECK: %[[C5:.*]] = arith.constant 5 : index
// CHECK: %[[A0:.*]] = tta.make_addr %arg0 to sizes: [4], strides: [1], offsets: [%[[C5]]], shape: [0], order: []
// CHECK: %[[V:.*]] = "tta.load"(%[[A0]])
// CHECK: %[[A1:.*]] = tta.make_addr %arg1 to sizes: [4], strides: [1], offsets: [0], shape: [0], order: []
// CHECK: "tta.store"(%[[A1]], %[[V]])
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

// -----

module {
// CHECK-LABEL: tt.func public @masked_2d_fallback(
// CHECK: %[[A0:.*]] = tta.make_addr %arg0 to sizes: [8]
// CHECK: %[[R0:.*]] = "tta.reindex"(%[[A0]], %{{.*}}, %{{.*}})
// CHECK: %[[L0:.*]] = "tta.load"(%[[R0]], %{{.*}})
// CHECK: %[[A1:.*]] = tta.make_addr %arg1 to sizes: [8]
// CHECK: %[[R1:.*]] = "tta.reindex"(%[[A1]], %{{.*}}, %{{.*}})
// CHECK: "tta.store"(%[[R1]], %[[L0]])
  tt.func public @masked_2d_fallback(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %col = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %row_exp = tt.expand_dims %row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %col_exp = tt.expand_dims %col {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %row_bcast = tt.broadcast %row_exp : tensor<2x1xi32> -> tensor<2x4xi32>
    %col_bcast = tt.broadcast %col_exp : tensor<1x4xi32> -> tensor<2x4xi32>
    %c4 = arith.constant 4 : i32
    %stride = tt.splat %c4 : i32 -> tensor<2x4xi32>
    %row_linear = arith.muli %row_bcast, %stride : tensor<2x4xi32>
    %offsets = arith.addi %row_linear, %col_bcast : tensor<2x4xi32>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_base, %offsets : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    %limit = tt.splat %arg2 : i32 -> tensor<2x4xi32>
    %mask = arith.cmpi slt, %offsets, %limit : tensor<2x4xi32>
    %zero = arith.constant 0.0 : f32
    %other = tt.splat %zero : f32 -> tensor<2x4xf32>
    %val = tt.load %in_ptrs, %mask, %other : tensor<2x4x!tt.ptr<f32>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %offsets : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    tt.store %out_ptrs, %val, %mask : tensor<2x4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func public @scalar_atomic_rmw_to_tta(
// CHECK: %[[A0:.*]] = "tta.atomic"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{kind = "add"}> : (!tta.addr<i32, 1, 1>, i32, i32, i1) -> i32
// CHECK: %[[A1:.*]] = "tta.atomic"(%{{.*}}, %{{.*}}, %{{.*}}) <{kind = "xchg"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
// CHECK: %[[A2:.*]] = "tta.atomic_cas"(%arg0, %{{.*}}, %{{.*}}, %{{.*}}) : (!tt.ptr<i32>, i32, i32, i32) -> i32
  tt.func public @scalar_atomic_rmw_to_tta(%arg0: !tt.ptr<i32>, %arg1: i32, %arg2: i32, %arg3: i1) {
    %a0 = tt.atomic_rmw add, acq_rel, gpu, %arg0, %arg1, %arg3 : (!tt.ptr<i32>, i32, i1) -> i32
    %a1 = tt.atomic_rmw exch, acq_rel, gpu, %arg0, %arg2 : (!tt.ptr<i32>, i32) -> i32
    %a2 = tt.atomic_cas acq_rel, gpu, %arg0, %arg1, %arg2 : (!tt.ptr<i32>, i32, i32) -> i32
    %sum0 = arith.addi %a0, %a1 : i32
    %sum = arith.addi %sum0, %a2 : i32
    %use = arith.addi %sum, %arg1 : i32
    %sink = arith.addi %use, %arg2 : i32
    tt.return
  }
}
