// RUN: triton-xyz-opt --triton-to-linalg-tta %s | FileCheck %s

// TODO

module {
// dim0 stays structured (2 rows with stride 4), dim1 is unstructured
// via indirect_reindex on column indices.
// CHECK-LABEL:   func.func @mixed_dim0_struct_dim1_unstruct(
// CHECK-SAME:      %[[SRC:[0-9a-zA-Z_$.]+]]: memref<*xf32>,
// CHECK-SAME:      %[[IDX:[0-9a-zA-Z_$.]+]]: memref<*xi32>,
// CHECK-SAME:      %[[DST:[0-9a-zA-Z_$.]+]]: memref<*xf32>) {
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C4:.*]] = arith.constant 4 : index
// CHECK:           %[[IDX_ALLOC:.*]] = memref.alloc() : memref<4xi32>
// CHECK:           %[[IDX_VIEW:.*]] = memref.reinterpret_cast %[[IDX]] to offset: [0], sizes: [4], strides: [1] : memref<*xi32> to memref<4xi32, strided<[1]>>
// CHECK:           memref.copy %[[IDX_VIEW]], %[[IDX_ALLOC]] : memref<4xi32, strided<[1]>> to memref<4xi32>
// CHECK:           %[[IDX_TENSOR:.*]] = bufferization.to_tensor %[[IDX_ALLOC]] restrict writable : memref<4xi32> to tensor<4xi32>
// CHECK:           %[[TMP:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           scf.for %[[IV:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK:             %[[IDX_ELEM:.*]] = tensor.extract %[[IDX_TENSOR]][%[[IV]]] : tensor<4xi32>
// CHECK:             %[[COL:.*]] = arith.index_cast %[[IDX_ELEM]] : i32 to index
// CHECK:             %[[SRC_COL:.*]] = memref.reinterpret_cast %[[SRC]] to offset: [%[[COL]]], sizes: [2, 1], strides: [4, 1] : memref<*xf32> to memref<2x1xf32, strided<[4, 1], offset: ?>>
// CHECK:             %[[TMP_COL:.*]] = memref.subview %[[TMP]][0, %[[IV]]] [2, 1] [1, 1] : memref<2x4xf32> to memref<2x1xf32, strided<[4, 1], offset: ?>>
// CHECK:             memref.copy %[[SRC_COL]], %[[TMP_COL]] : memref<2x1xf32, strided<[4, 1], offset: ?>> to memref<2x1xf32, strided<[4, 1], offset: ?>>
// CHECK:           }
// CHECK:           %[[VAL:.*]] = bufferization.to_tensor %[[TMP]] restrict writable : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:           scf.for %[[JV:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK:             %[[IDX_ELEM_2:.*]] = tensor.extract %[[IDX_TENSOR]][%[[JV]]] : tensor<4xi32>
// CHECK:             %[[COL_2:.*]] = arith.index_cast %[[IDX_ELEM_2]] : i32 to index
// CHECK:             %[[DST_COL:.*]] = memref.reinterpret_cast %[[DST]] to offset: [%[[COL_2]]], sizes: [2, 1], strides: [4, 1] : memref<*xf32> to memref<2x1xf32, strided<[4, 1], offset: ?>>
// CHECK:             %[[SLICE:.*]] = tensor.extract_slice %[[VAL]][0, %[[JV]]] [2, 1] [1, 1] : tensor<2x4xf32> to tensor<2x1xf32>
// CHECK:             bufferization.materialize_in_destination %[[SLICE]] in writable %[[DST_COL]] : (tensor<2x1xf32>, memref<2x1xf32, strided<[4, 1], offset: ?>>) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }
  tt.func public @mixed_dim0_struct_dim1_unstruct(%src: !tt.ptr<f32>, %idx: !tt.ptr<i32>, %dst: !tt.ptr<f32>) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %idx_base = tt.splat %idx : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %idx_ptrs = tt.addptr %idx_base, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %col_idx = tt.load %idx_ptrs : tensor<4x!tt.ptr<i32>>

    %src_addr = tta.make_addr %src to sizes: [2, 4], strides: [4, 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
    %src_idx = "tta.indirect_reindex"(%src_addr, %col_idx) <{indirect_dim = 1 : i32}> : (!tta.addr<f32, 2, 1>, tensor<4xi32>) -> !tta.addr<f32, 2, 1>
    %other = arith.constant 0.0 : f32
    %val = "tta.load"(%src_idx, %other) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, f32) -> tensor<2x4xf32>

    %dst_addr = tta.make_addr %dst to sizes: [2, 4], strides: [4, 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
    %dst_idx = "tta.indirect_reindex"(%dst_addr, %col_idx) <{indirect_dim = 1 : i32}> : (!tta.addr<f32, 2, 1>, tensor<4xi32>) -> !tta.addr<f32, 2, 1>
    "tta.store"(%dst_idx, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<2x4xf32>) -> ()
    tt.return
  }
}
