// RUN: triton-xyz-opt --split-input-file --triton-to-tta-structured %s | FileCheck %s

module {
// CHECK-LABEL: tt.func @fallback_mask_rank_not_1d(
// CHECK: tt.load %{{.*}}, %{{.*}}, %{{.*}} {tta.fallback, tta.fallback_reason = "mask_rank_not_1d"}
// CHECK: tt.store %{{.*}}, %{{.*}}, %{{.*}} {tta.fallback, tta.fallback_reason = "mask_rank_not_1d"}
  tt.func @fallback_mask_rank_not_1d(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
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
// CHECK-LABEL: tt.func @fallback_mask_analysis_failed(
// CHECK: tt.load %{{.*}}, %{{.*}}, %{{.*}} {tta.fallback, tta.fallback_reason = "mask_analysis_failed"}
// CHECK: tt.store %{{.*}}, %{{.*}}, %{{.*}} {tta.fallback, tta.fallback_reason = "mask_analysis_failed"}
  tt.func @fallback_mask_analysis_failed(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %out_ptrs = tt.addptr %out_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %limit = tt.splat %arg2 : i32 -> tensor<4xi32>
    %mask = arith.cmpi eq, %range, %limit : tensor<4xi32>
    %zero = arith.constant 0.0 : f32
    %other = tt.splat %zero : f32 -> tensor<4xf32>
    %val = tt.load %in_ptrs, %mask, %other : tensor<4x!tt.ptr<f32>>
    tt.store %out_ptrs, %val, %mask : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @fallback_other_not_scalar_splat(
// CHECK: tt.load %{{.*}}, %{{.*}}, %{{.*}} {tta.fallback, tta.fallback_reason = "other_not_scalar_splat"}
  tt.func @fallback_other_not_scalar_splat(%arg0: !tt.ptr<f32>, %arg1: i32) -> tensor<4xf32> {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %limit = tt.splat %arg1 : i32 -> tensor<4xi32>
    %mask = arith.cmpi slt, %range, %limit : tensor<4xi32>
    %other = arith.constant dense<[0.0, 1.0, 2.0, 3.0]> : tensor<4xf32>
    %val = tt.load %ptrs, %mask, %other : tensor<4x!tt.ptr<f32>>
    tt.return %val : tensor<4xf32>
  }
}

// -----

module {
// CHECK-LABEL: tt.func @fallback_boundary_check_not_supported(
// CHECK: tt.load %{{.*}} {boundaryCheck = array<i32: 0>, padding = 2 : i32, tta.fallback, tta.fallback_reason = "boundary_check_not_supported"}
// CHECK: tt.store %{{.*}}, %{{.*}} {boundaryCheck = array<i32: 0>, tta.fallback, tta.fallback_reason = "boundary_check_not_supported"}
  tt.func @fallback_boundary_check_not_supported(%arg0: !tt.ptr<f32>) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %val = tt.load %ptrs {boundaryCheck = array<i32: 0>, padding = 2 : i32} : tensor<4x!tt.ptr<f32>>
    tt.store %ptrs, %val {boundaryCheck = array<i32: 0>} : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @preserve_premarked_load_store(
// CHECK: tt.load %{{.*}} {tta.fallback, tta.fallback_reason = "pre_marked_load"}
// CHECK: tt.store %{{.*}}, %{{.*}} {tta.fallback, tta.fallback_reason = "pre_marked_store"}
  tt.func @preserve_premarked_load_store(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %out_ptrs = tt.addptr %out_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %val = tt.load %in_ptrs {tta.fallback, tta.fallback_reason = "pre_marked_load"} : tensor<4x!tt.ptr<f32>>
    tt.store %out_ptrs, %val {tta.fallback, tta.fallback_reason = "pre_marked_store"} : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @fallback_tensor_ptr_unhandled(
// CHECK: tt.make_tensor_ptr %arg0{{.*}} {order = array<i32: 1, 0>, tta.fallback, tta.fallback_reason = "tensor_ptr_unhandled"}
  tt.func @fallback_tensor_ptr_unhandled(%arg0: !tt.ptr<f16>) -> !tt.ptr<tensor<4x4xf16>> {
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %ptr = tt.make_tensor_ptr %arg0, [%c4_i64, %c4_i64], [%c4_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<4x4xf16>>
    tt.return %ptr : !tt.ptr<tensor<4x4xf16>>
  }
}

// -----

module {
// CHECK-LABEL: tt.func @preserve_premarked_tensor_ptr_reason(
// CHECK: tt.make_tensor_ptr %arg0{{.*}} {order = array<i32: 1, 0>, tta.fallback, tta.fallback_reason = "pre_marked_reason"}
  tt.func @preserve_premarked_tensor_ptr_reason(%arg0: !tt.ptr<f16>) -> !tt.ptr<tensor<4x4xf16>> {
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %ptr = tt.make_tensor_ptr %arg0, [%c4_i64, %c4_i64], [%c4_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>, tta.fallback, tta.fallback_reason = "pre_marked_reason"} : <tensor<4x4xf16>>
    tt.return %ptr : !tt.ptr<tensor<4x4xf16>>
  }
}

// -----

module {
// CHECK-LABEL: tt.func @override_premarked_tensor_ptr_reason(
// CHECK: tt.make_tensor_ptr %arg0{{.*}} {order = array<i32: 1, 0>, tta.fallback, tta.fallback_reason = "tensor_ptr_unhandled"}
  tt.func @override_premarked_tensor_ptr_reason(%arg0: !tt.ptr<f16>) -> !tt.ptr<tensor<4x4xf16>> {
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %ptr = tt.make_tensor_ptr %arg0, [%c4_i64, %c4_i64], [%c4_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>, tta.fallback_reason = "preexisting_reason"} : <tensor<4x4xf16>>
    tt.return %ptr : !tt.ptr<tensor<4x4xf16>>
  }
}
