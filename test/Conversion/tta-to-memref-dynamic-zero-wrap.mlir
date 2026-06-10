// RUN: triton-xyz-opt --split-input-file --tta-normalize --tta-to-memref --canonicalize --cse %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @dynamic_stride_scaled_wrap_boundary(
// CHECK-SAME:      %[[ARG0:[-0-9A-Za-z$._]+]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[-0-9A-Za-z$._]+]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[-0-9A-Za-z$._]+]]: index,
// CHECK-SAME:      %[[ARG3:[-0-9A-Za-z$._]+]]: index) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[ARG3]], %[[ARG2]] : index
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           linalg.fill ins(%[[CONSTANT_2]] : f32) outs(%[[ALLOC_0]] : memref<4xf32>)
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xf32> to memref<?xf32>
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_1]] step %[[CONSTANT_0]] {
// CHECK:             %[[MULI_1:.*]] = arith.muli %[[VAL_0]], %[[ARG2]] : index
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi sgt, %[[MULI_0]], %[[CONSTANT_3]] : index
// CHECK:             %[[CMPI_1:.*]] = arith.cmpi eq, %[[ARG2]], %[[CONSTANT_3]] : index
// CHECK:             %[[ORI_0:.*]] = arith.ori %[[CMPI_0]], %[[CMPI_1]] : i1
// CHECK:             cf.assert %[[ORI_0]], "tta-to-memref: wrap boundary must be > 0 unless stride is 0"
// CHECK:             %[[SELECT_0:.*]] = arith.select %[[CMPI_1]], %[[CONSTANT_0]], %[[MULI_0]] : index
// CHECK:             %[[REMSI_0:.*]] = arith.remsi %[[MULI_1]], %[[SELECT_0]] : index
// CHECK:             %[[CMPI_2:.*]] = arith.cmpi slt, %[[REMSI_0]], %[[CONSTANT_3]] : index
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[REMSI_0]], %[[SELECT_0]] : index
// CHECK:             %[[SELECT_1:.*]] = arith.select %[[CMPI_2]], %[[ADDI_0]], %[[REMSI_0]] : index
// CHECK:             %[[SELECT_2:.*]] = arith.select %[[CMPI_1]], %[[MULI_1]], %[[SELECT_1]] : index
// CHECK:             %[[LOAD_0:.*]] = memref.load %[[CAST_0]]{{\[}}%[[SELECT_2]]] : memref<?xf32>
// CHECK:             memref.store %[[LOAD_0]], %[[ALLOC_0]]{{\[}}%[[VAL_0]]] : memref<4xf32>
// CHECK:           }
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
// CHECK:           bufferization.materialize_in_destination %[[TO_TENSOR_0]] in writable %[[REINTERPRET_CAST_0]] : (tensor<4xf32>, memref<4xf32, strided<[1]>>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @dynamic_stride_scaled_wrap_boundary(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %stride: index, %size: index) {
    %c0 = arith.constant 0 : index
    %boundary = arith.muli %size, %stride : index
    %addr = tta.make_addr %src to sizes: [4], strides: [%stride], offsets: [%c0], wrap_boundaries: [%boundary], layout: "strided" : <f32> to !tta.addr<f32, 1, 1>
    %other = arith.constant 0.0 : f32
    %val = "tta.load"(%addr, %other) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, f32) -> tensor<4xf32>
    %dst_addr = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], wrap_boundaries: [0], layout: "strided" : <f32> to !tta.addr<f32, 1, 1>
    "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
    tt.return
  }
}
