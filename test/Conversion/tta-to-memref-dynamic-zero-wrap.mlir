// RUN: triton-xyz-opt --split-input-file --tta-normalize --tta-to-memref --canonicalize --cse %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @dynamic_stride_scaled_wrap_boundary(
// CHECK-SAME:        %[[SRC:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:        %[[DST:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:        %[[STRIDE:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: index,
// CHECK-SAME:        %[[SIZE:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: index) {
// CHECK-DAG:       %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[ONE:.*]] = arith.constant 1 : index
// CHECK:           %[[BOUNDARY:.*]] = arith.muli %[[SIZE]], %[[STRIDE]] : index
// CHECK:           %[[POSITIVE:.*]] = arith.cmpi sgt, %[[BOUNDARY]], %[[ZERO]] : index
// CHECK:           %[[ZERO_STRIDE:.*]] = arith.cmpi eq, %[[STRIDE]], %[[ZERO]] : index
// CHECK:           %[[VALID:.*]] = arith.ori %[[POSITIVE]], %[[ZERO_STRIDE]] : i1
// CHECK:           cf.assert %[[VALID]], "tta-to-memref: wrap boundary must be > 0 unless stride is 0"
// CHECK:           %[[SAFE_BOUNDARY:.*]] = arith.select %[[ZERO_STRIDE]], %[[ONE]], %[[BOUNDARY]] : index
// CHECK:           %[[REM:.*]] = arith.remsi {{.*}}, %[[SAFE_BOUNDARY]] : index
// CHECK:           %[[WRAPPED:.*]] = arith.select {{.*}}, {{.*}}, %[[REM]] : index
// CHECK:           arith.select %[[ZERO_STRIDE]], {{.*}}, %[[WRAPPED]] : index
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
