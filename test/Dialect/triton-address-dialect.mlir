// RUN: triton-xyz-opt --split-input-file %s | FileCheck %s

module {
// CHECK-LABEL: tt.func @address_core_ops(
  tt.func @address_core_ops(%base: !tt.ptr<f32>, %off: i32, %ibase: !tt.ptr<i32>, %ioff: i32) {
    %idx = arith.constant dense<[0, 1]> : tensor<2xi32>
    %mask = arith.constant dense<[true, false]> : tensor<2xi1>
    %smask = arith.constant true
    %val = arith.constant 1.000000e+00 : f32
    %other = arith.constant 0.000000e+00 : f32
    %ival = arith.constant 1 : i32

    // CHECK: tta.make_addr
    %addr = tta.make_addr %base to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>

    // CHECK: "tta.reindex"(%
    %r0 = "tta.reindex"(%addr) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 1>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2x!tt.ptr<f32>>

    // CHECK: "tta.reindex"(%
    %r1 = "tta.reindex"(%r0, %idx, %mask) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0, 0>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2xi32>, tensor<2xi1>) -> tensor<2x2x!tt.ptr<f32>>

    // CHECK: "tta.advance"(%
    %r2 = "tta.advance"(%r1) <{static_deltas = array<i64: 1, 0>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2x!tt.ptr<f32>>

    // CHECK: "tta.load"(%
    %loaded = "tta.load"(%r2) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
    // CHECK: "tta.load"(%
    %loadedOther = "tta.load"(%r2, %other) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, f32) -> tensor<2x2xf32>
    // CHECK: "tta.store"(%
    "tta.store"(%r2, %loaded) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()

    // CHECK: "tta.atomic"(%
    %a0 = "tta.atomic"(%base, %off, %val) <{kind = "add"}> : (!tt.ptr<f32>, i32, f32) -> f32
    // CHECK: "tta.atomic"(%
    %a1 = "tta.atomic"(%base, %off, %a0, %smask) <{kind = "fadd"}> : (!tt.ptr<f32>, i32, f32, i1) -> f32
    // CHECK: "tta.atomic"(%
    %a2 = "tta.atomic"(%ibase, %ioff, %ival) <{kind = "xor"}> : (!tt.ptr<i32>, i32, i32) -> i32

    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @address_mixed_dynamic_make_addr(
  tt.func @address_mixed_dynamic_make_addr(%base: !tt.ptr<f32>, %s0: index, %o0: index) {
    // CHECK: tta.make_addr
    %addr = tta.make_addr %base to sizes: [2, 2], strides: [%s0, 1], offsets: [%o0, 0], shape: [2, %s0], order: [1, 0] : <f32> to !tt.ptr<tensor<2x2xf32>>
    // CHECK: "tta.reindex"(%
    %r = "tta.reindex"(%addr) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>}> : (!tt.ptr<tensor<2x2xf32>>) -> !tt.ptr<tensor<2x2xf32>>
    // CHECK: "tta.advance"(%
    %a = "tta.advance"(%r) <{static_deltas = array<i64: 0, 1>}> : (!tt.ptr<tensor<2x2xf32>>) -> !tt.ptr<tensor<2x2xf32>>
    // CHECK: "tta.load"(%
    %v = "tta.load"(%a) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<2x2xf32>>) -> tensor<2x2xf32>
    // CHECK: "tta.store"(%
    "tta.store"(%a, %v) <{static_mask_dims = array<i64>}> : (!tt.ptr<tensor<2x2xf32>>, tensor<2x2xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @address_scalar_ptr_reindex_advance(
  tt.func @address_scalar_ptr_reindex_advance(%base: !tt.ptr<f32>) {
    %off = arith.constant 0 : i32
    %val = arith.constant 2.000000e+00 : f32
    // CHECK: "tta.reindex"(%
    %r0 = "tta.reindex"(%base) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>}> : (!tt.ptr<f32>) -> !tt.ptr<f32>
    // CHECK: "tta.advance"(%
    %r1 = "tta.advance"(%r0) <{static_deltas = array<i64: 1>}> : (!tt.ptr<f32>) -> !tt.ptr<f32>
    // CHECK: "tta.atomic"(%
    %a = "tta.atomic"(%r1, %off, %val) <{kind = "xchg"}> : (!tt.ptr<f32>, i32, f32) -> f32
    tt.return
  }
}
