// RUN: triton-xyz-opt --split-input-file --triton-to-linalg-tta %s | FileCheck %s

// TODO: rm this test

// CHECK-NOT: tta.
// CHECK-NOT: !tta.addr

module {
  // CHECK-LABEL: func.func @loop_ptr_iter_args_lowering(
  // CHECK-SAME: memref<*xf32>, %[[ARG1:.*]]: memref<*xf32>, %[[N:.*]]: i32
  // CHECK: %[[LOOP:.*]]:2 = scf.for %{{.*}} = %{{.*}} to %[[N]] step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (tensor<4xi32>, tensor<4xi32>)
  // CHECK: memref.reinterpret_cast %arg0
  // CHECK: memref.copy
  // CHECK: memref.reinterpret_cast %[[ARG1]]
  // CHECK: bufferization.materialize_in_destination
  // CHECK: scf.yield %{{.*}}, %{{.*}} : tensor<4xi32>, tensor<4xi32>
  tt.func public @loop_ptr_iter_args_lowering(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %out_ptrs = tt.addptr %out_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>

    %res:2 = scf.for %iv = %c0_i32 to %arg2 step %c1_i32 iter_args(%in = %in_ptrs, %out = %out_ptrs) -> (tensor<4x!tt.ptr<f32>>, tensor<4x!tt.ptr<f32>>) : i32 {
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

// TODO: fix `tt.load` with tta pipeline

module {
  // CHECK-LABEL: func.func @if_ptr_merge_lowering(
  // CHECK-SAME: memref<*xf32>, %[[ARG1:.*]]: memref<*xf32>, %[[PRED:.*]]: i1
  // CHECK: %[[SEL:.*]] = arith.select %[[PRED]], %{{.*}}, %{{.*}} : tensor<4x!tt.ptr<f32>>
  // CHECK: %[[LOOP:.*]] = scf.for
  // CHECK: tt.load
  // CHECK: memref.reinterpret_cast %[[ARG1]]
  // CHECK: bufferization.materialize_in_destination
  tt.func public @if_ptr_merge_lowering(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %pred: i1) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %in0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %in1 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptr0 = tt.addptr %in0, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %ptr1 = tt.addptr %in1, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>

    %chosen = scf.if %pred -> (tensor<4x!tt.ptr<f32>>) {
      scf.yield %ptr0 : tensor<4x!tt.ptr<f32>>
    } else {
      scf.yield %ptr1 : tensor<4x!tt.ptr<f32>>
    }

    %v = tt.load %chosen : tensor<4x!tt.ptr<f32>>
    %out = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %out_ptrs, %v : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}
