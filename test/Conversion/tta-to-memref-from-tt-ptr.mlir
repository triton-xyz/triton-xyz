// RUN: triton-xyz-opt --split-input-file --tta-normalize --tta-to-memref --canonicalize --cse %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @from_tt_ptr_indirect_load_store(
// CHECK-SAME:      %[[ARG0:[-0-9A-Za-z$._]+]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[-0-9A-Za-z$._]+]]: !tt.ptr<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xindex>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 4 : index
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[CONSTANT_0]]{{\[}}%[[VAL_0]]] : tensor<4xindex>
// CHECK:             %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[EXTRACT_0]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_0]]{{\[}}%[[VAL_0]]] [1] [1] : memref<4xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             memref.copy %[[REINTERPRET_CAST_0]], %[[SUBVIEW_0]] : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:           }
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK:             %[[EXTRACT_1:.*]] = tensor.extract %[[CONSTANT_0]]{{\[}}%[[VAL_1]]] : tensor<4xindex>
// CHECK:             %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: {{\[}}%[[EXTRACT_1]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[TO_TENSOR_0]]{{\[}}%[[VAL_1]]] [1] [1] : tensor<4xf32> to tensor<1xf32>
// CHECK:             bufferization.materialize_in_destination %[[EXTRACT_SLICE_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @from_tt_ptr_indirect_load_store(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %range = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %src_ptrs = tt.addptr %src_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %src_addr = tta.from_tt_ptr %src_ptrs : tensor<4x!tt.ptr<f32>> to !tta.addr<f32, 1, 1>
    %val = "tta.load"(%src_addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>

    %dst_base = tt.splat %dst : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %dst_ptrs = tt.addptr %dst_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %dst_addr = tta.from_tt_ptr %dst_ptrs : tensor<4x!tt.ptr<f32>> to !tta.addr<f32, 1, 1>
    "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
    tt.return
  }
}
