// RUN: triton-xyz-opt --split-input-file --tta-to-memref --canonicalize --cse %s | FileCheck %s

// CHECK-NOT: tta.
// CHECK-NOT: tts.

module {
// CHECK-LABEL: tt.func @structured_basic(
// CHECK: memref.reinterpret_cast
// CHECK: bufferization.materialize_in_destination
  tt.func @structured_basic(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %src_addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
    %val = "tta.load"(%src_addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>) -> tensor<4xf32>
    %dst_addr = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
    "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @block_ptr_with_advance(
// CHECK: memref.reinterpret_cast
// CHECK: memref<4x4xf16
// CHECK: bufferization.materialize_in_destination
  tt.func @block_ptr_with_advance(%base: !tt.ptr<f16>) {
    %addr = tta.make_addr %base to sizes: [4, 4], strides: [4, 1], offsets: [0, 0], shape: [4, 4], order: [1, 0] : <f16> to !tt.ptr<tensor<4x4xf16>>
    %next = "tta.advance"(%addr) <{static_deltas = array<i64: 0, 1>}> : (!tt.ptr<tensor<4x4xf16>>) -> !tt.ptr<tensor<4x4xf16>>
    %val = "tta.load"(%next) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<4x4xf16>>) -> tensor<4x4xf16>
    "tta.store"(%addr, %val) <{static_mask_dims = array<i64>}> : (!tt.ptr<tensor<4x4xf16>>, tensor<4x4xf16>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @indirect_masked_load_store(
// CHECK: scf.for
// CHECK: tensor.extract
// CHECK: bufferization.materialize_in_destination
  tt.func @indirect_masked_load_store(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %mask = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
    %other = arith.constant 1.250000e+00 : f32

    %src_addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
    %src_idx = "tta.reindex"(%src_addr, %offsets, %mask) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0>}> : (tensor<4x!tt.ptr<f32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4x!tt.ptr<f32>>
    %val = "tta.load"(%src_idx, %other) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>, f32) -> tensor<4xf32>

    %dst_addr = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
    %dst_idx = "tta.reindex"(%dst_addr, %offsets, %mask) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0>}> : (tensor<4x!tt.ptr<f32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4x!tt.ptr<f32>>
    "tta.store"(%dst_idx, %val) <{static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>) -> ()
    tt.return
  }
}
