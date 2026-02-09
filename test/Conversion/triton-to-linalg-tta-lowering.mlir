// RUN: triton-xyz-opt --split-input-file --triton-to-linalg-tta %s | FileCheck %s

// CHECK-NOT: tta.
// CHECK-NOT: tts.

module {
// CHECK-LABEL: func.func @vector_add(
// CHECK-SAME: memref<*xf32>
// CHECK: memref.reinterpret_cast
// CHECK: linalg.generic
// CHECK: bufferization.materialize_in_destination
  tt.func @vector_add(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %lhs_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %rhs_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_base = tt.splat %arg2 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %lhs_ptrs = tt.addptr %lhs_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %rhs_ptrs = tt.addptr %rhs_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %out_ptrs = tt.addptr %out_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %lhs = tt.load %lhs_ptrs : tensor<4x!tt.ptr<f32>>
    %rhs = tt.load %rhs_ptrs : tensor<4x!tt.ptr<f32>>
    %sum = arith.addf %lhs, %rhs : tensor<4xf32>
    tt.store %out_ptrs, %sum : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: func.func @gather_scatter_2d(
// CHECK-SAME: memref<*xf32>
// CHECK-SAME: memref<*xi32>
// CHECK: scf.for
// CHECK: tensor.extract
// CHECK: memref.copy
// CHECK: bufferization.materialize_in_destination
  tt.func @gather_scatter_2d(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<f32>) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %idx_base = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %idx_ptrs = tt.addptr %idx_base, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %idx = tt.load %idx_ptrs : tensor<4x!tt.ptr<i32>>
    %idx_row = tt.expand_dims %idx {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %c4_i32 = arith.constant 4 : i32
    %stride = tt.splat %c4_i32 : i32 -> tensor<4x1xi32>
    %row_offsets = arith.muli %idx_row, %stride : tensor<4x1xi32>
    %col = tt.expand_dims %range {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %row_bcast = tt.broadcast %row_offsets : tensor<4x1xi32> -> tensor<4x4xi32>
    %col_bcast = tt.broadcast %col : tensor<1x4xi32> -> tensor<4x4xi32>
    %offsets = arith.addi %row_bcast, %col_bcast : tensor<4x4xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %offsets : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %val = tt.load %ptrs : tensor<4x4x!tt.ptr<f32>>
    %row = tt.expand_dims %range {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %row_stride = tt.splat %c4_i32 : i32 -> tensor<4x1xi32>
    %row_linear = arith.muli %row, %row_stride : tensor<4x1xi32>
    %row_linear_bcast = tt.broadcast %row_linear : tensor<4x1xi32> -> tensor<4x4xi32>
    %linear_offsets = arith.addi %row_linear_bcast, %col_bcast : tensor<4x4xi32>
    %out_base = tt.splat %arg2 : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %linear_offsets : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    tt.store %out_ptrs, %val : tensor<4x4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: func.func @atomic_scalar_tta_route(
// CHECK-SAME: memref<*xi32>
// CHECK: memref.generic_atomic_rmw
// CHECK: arith.select
// CHECK: memref.atomic_yield
  tt.func @atomic_scalar_tta_route(%arg0: !tt.ptr<i32>, %arg1: i32, %arg2: i1) {
    %r = tt.atomic_rmw add, acq_rel, gpu, %arg0, %arg1, %arg2 : (!tt.ptr<i32>, i32, i1) -> i32
    %u = arith.addi %r, %arg1 : i32
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: func.func @atomic_cas_scalar_tta_route(
// CHECK-SAME: memref<*xi32>
// CHECK: memref.generic_atomic_rmw
// CHECK: arith.cmpi eq
// CHECK: arith.select
// CHECK: memref.atomic_yield
  tt.func @atomic_cas_scalar_tta_route(%arg0: !tt.ptr<i32>, %arg1: i32, %arg2: i32) {
    %r = tt.atomic_cas acq_rel, gpu, %arg0, %arg1, %arg2 : (!tt.ptr<i32>, i32, i32) -> i32
    %u = arith.addi %r, %arg2 : i32
    tt.return
  }
}
