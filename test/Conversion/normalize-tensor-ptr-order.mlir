// RUN: triton-xyz-opt --split-input-file --normalize-tensor-ptr-order %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @normalize_load_store(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f16>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 8 : index
// CHECK:           %[[MAKE_TPTR_0:.*]] = tts.make_tptr %[[ARG0]] to sizes: [8, 4], strides: {{\[}}%[[CONSTANT_1]], %[[CONSTANT_3]]], offsets: {{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]], shape: {{\[}}%[[CONSTANT_3]], %[[CONSTANT_2]]], order: [1, 0] : <f16> to !tt.ptr<tensor<8x4xf16>>
// CHECK:           %[[VAL_0:.*]] = "tts.load"(%[[MAKE_TPTR_0]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<8x4xf16>>) -> tensor<8x4xf16>
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4x8xf16>
// CHECK:           %[[TRANSPOSE_0:.*]] = linalg.transpose ins(%[[VAL_0]] : tensor<8x4xf16>) outs(%[[EMPTY_0]] : tensor<4x8xf16>) permutation = [1, 0]
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<8x4xf16>
// CHECK:           %[[TRANSPOSE_1:.*]] = linalg.transpose ins(%[[TRANSPOSE_0]] : tensor<4x8xf16>) outs(%[[EMPTY_1]] : tensor<8x4xf16>) permutation = [1, 0]
// CHECK:           "tts.store"(%[[MAKE_TPTR_0]], %[[TRANSPOSE_1]]) <{static_mask_dims = array<i64>}> : (!tt.ptr<tensor<8x4xf16>>, tensor<8x4xf16>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @normalize_load_store(%base: !tt.ptr<f16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %tptr = tts.make_tptr %base to sizes: [4, 8], strides: [%c8, %c1], offsets: [%c0, %c0], shape: [%c4, %c8], order: [0, 1] : <f16> to !tt.ptr<tensor<4x8xf16>>
    %val = "tts.load"(%tptr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<4x8xf16>>) -> tensor<4x8xf16>
    "tts.store"(%tptr, %val) <{static_mask_dims = array<i64>}> : (!tt.ptr<tensor<4x8xf16>>, tensor<4x8xf16>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @normalize_advance(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f16>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 8 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 2 : i32
// CHECK:           %[[MAKE_TPTR_0:.*]] = tts.make_tptr %[[ARG0]] to sizes: [8, 4], strides: {{\[}}%[[CONSTANT_1]], %[[CONSTANT_3]]], offsets: {{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]], shape: {{\[}}%[[CONSTANT_3]], %[[CONSTANT_2]]], order: [1, 0] : <f16> to !tt.ptr<tensor<8x4xf16>>
// CHECK:           %[[ADVANCE_0:.*]] = tt.advance %[[MAKE_TPTR_0]], {{\[}}%[[CONSTANT_5]], %[[CONSTANT_4]]] : <tensor<8x4xf16>>
// CHECK:           %[[VAL_0:.*]] = "tts.load"(%[[ADVANCE_0]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<8x4xf16>>) -> tensor<8x4xf16>
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4x8xf16>
// CHECK:           %[[TRANSPOSE_0:.*]] = linalg.transpose ins(%[[VAL_0]] : tensor<8x4xf16>) outs(%[[EMPTY_0]] : tensor<4x8xf16>) permutation = [1, 0]
// CHECK:           tt.return
// CHECK:         }
  tt.func @normalize_advance(%base: !tt.ptr<f16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %tptr = tts.make_tptr %base to sizes: [4, 8], strides: [%c8, %c1], offsets: [%c0, %c0], shape: [%c4, %c8], order: [0, 1] : <f16> to !tt.ptr<tensor<4x8xf16>>
    %adv = tt.advance %tptr, [%c1_i32, %c2_i32] : <tensor<4x8xf16>>
    %val = "tts.load"(%adv) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<4x8xf16>>) -> tensor<4x8xf16>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @normalize_loop(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f16>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 8 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_6:.*]] = arith.constant 2 : i32
// CHECK:           %[[MAKE_TPTR_0:.*]] = tts.make_tptr %[[ARG0]] to sizes: [8, 4], strides: {{\[}}%[[CONSTANT_1]], %[[CONSTANT_3]]], offsets: {{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]], shape: {{\[}}%[[CONSTANT_3]], %[[CONSTANT_2]]], order: [1, 0] : <f16> to !tt.ptr<tensor<8x4xf16>>
// CHECK:           %[[FOR_0:.*]] = scf.for %[[VAL_0:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_6]] step %[[CONSTANT_5]] iter_args(%[[VAL_1:.*]] = %[[MAKE_TPTR_0]]) -> (!tt.ptr<tensor<8x4xf16>>)  : i32 {
// CHECK:             %[[VAL_2:.*]] = "tts.load"(%[[VAL_1]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<8x4xf16>>) -> tensor<8x4xf16>
// CHECK:             %[[EMPTY_0:.*]] = tensor.empty() : tensor<4x8xf16>
// CHECK:             %[[TRANSPOSE_0:.*]] = linalg.transpose ins(%[[VAL_2]] : tensor<8x4xf16>) outs(%[[EMPTY_0]] : tensor<4x8xf16>) permutation = [1, 0]
// CHECK:             %[[ADVANCE_0:.*]] = tt.advance %[[VAL_1]], {{\[}}%[[CONSTANT_6]], %[[CONSTANT_5]]] : <tensor<8x4xf16>>
// CHECK:             scf.yield %[[ADVANCE_0]] : !tt.ptr<tensor<8x4xf16>>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @normalize_loop(%base: !tt.ptr<f16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %tptr = tts.make_tptr %base to sizes: [4, 8], strides: [%c8, %c1], offsets: [%c0, %c0], shape: [%c4, %c8], order: [0, 1] : <f16> to !tt.ptr<tensor<4x8xf16>>
    %res = scf.for %iv = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg = %tptr) -> (!tt.ptr<tensor<4x8xf16>>) : i32 {
      %val = "tts.load"(%arg) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<4x8xf16>>) -> tensor<4x8xf16>
      %next = tt.advance %arg, [%c1_i32, %c2_i32] : <tensor<4x8xf16>>
      scf.yield %next : !tt.ptr<tensor<4x8xf16>>
    }
    tt.return
  }
}
