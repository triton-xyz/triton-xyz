// RUN: triton-xyz-opt --split-input-file --tta-to-xyz %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @indirect_load_store_to_xyz(
// CHECK-DAG:       %[[OTHER:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       %[[IDX:.*]] = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi32>
// CHECK:           %[[SRC_ADDR:.*]] = tta.make_addr
// CHECK:           %[[VAL:.*]] = "xyz.gather"(%[[SRC_ADDR]], %[[IDX]], %[[OTHER]]) <{gather_dim = 1 : i32
// CHECK:           %[[DST_ADDR:.*]] = tta.make_addr
// CHECK:           "xyz.scatter"(%[[DST_ADDR]], %[[IDX]], %[[VAL]]) <{gather_dim = 1 : i32
// CHECK:           tt.return
  tt.func @indirect_load_store_to_xyz(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %idx = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi32>

    %src_addr = tta.make_addr %src to sizes: [2, 4], strides: [4, 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
    %src_idx = "tta.indirect_reindex"(%src_addr, %idx) <{indirect_dim = 1 : i32}> : (!tta.addr<f32, 2, 1>, tensor<4xi32>) -> !tta.addr<f32, 2, 1>
    %other = arith.constant 0.0 : f32
    %val = "tta.load"(%src_idx, %other) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, f32) -> tensor<2x4xf32>

    %dst_addr = tta.make_addr %dst to sizes: [2, 4], strides: [4, 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
    %dst_idx = "tta.indirect_reindex"(%dst_addr, %idx) <{indirect_dim = 1 : i32}> : (!tta.addr<f32, 2, 1>, tensor<4xi32>) -> !tta.addr<f32, 2, 1>
    "tta.store"(%dst_idx, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<2x4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @direct_load_unchanged(
// CHECK:           %[[ADDR:.*]] = tta.make_addr
// CHECK:           %[[VAL:.*]] = "tta.load"(%[[ADDR]])
// CHECK-NOT:       "xyz.gather"
// CHECK:           tt.return %[[VAL]]
  tt.func @direct_load_unchanged(%src: !tt.ptr<f32>) -> tensor<4xf32> {
    %addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %v = "tta.load"(%addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
    tt.return %v : tensor<4xf32>
  }
}
