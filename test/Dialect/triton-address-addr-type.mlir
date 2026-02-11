// RUN: triton-xyz-opt --split-input-file %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @addr_type_roundtrip(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tta.addr<f32, 1, 1>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tta.addr<i32, 2, 3>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tta.addr<bf16, 3, 0>) -> (!tta.addr<f32, 1, 1>, !tta.addr<i32, 2, 3>, !tta.addr<bf16, 3, 0>) {
// CHECK:           tt.return %[[ARG0]], %[[ARG1]], %[[ARG2]] : !tta.addr<f32, 1, 1>, !tta.addr<i32, 2, 3>, !tta.addr<bf16, 3, 0>
// CHECK:         }
  tt.func @addr_type_roundtrip(%a0: !tta.addr<f32, 1, 1>,
                               %a1: !tta.addr<i32, 2, 3>,
                               %a2: !tta.addr<bf16, 3, 0>)
      -> (!tta.addr<f32, 1, 1>, !tta.addr<i32, 2, 3>, !tta.addr<bf16, 3, 0>) {
    tt.return %a0, %a1, %a2 : !tta.addr<f32, 1, 1>, !tta.addr<i32, 2, 3>, !tta.addr<bf16, 3, 0>
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @from_scalar_ptr(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[FROM_TT_PTR_0:.*]] = tta.from_tt_ptr %[[ARG0]] : !tt.ptr<f32> to !tta.addr<f32, 1, 1>
// CHECK:           tt.return
// CHECK:         }
  tt.func @from_scalar_ptr(%arg0: !tt.ptr<f32>) {
    %0 = tta.from_tt_ptr %arg0 : !tt.ptr<f32> to !tta.addr<f32, 1, 1>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @from_ptr_tensor(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4x!tt.ptr<f16>>) {
// CHECK:           %[[FROM_TT_PTR_0:.*]] = tta.from_tt_ptr %[[ARG0]] : tensor<4x!tt.ptr<f16>> to !tta.addr<f16, 1, 1>
// CHECK:           tt.return
// CHECK:         }
  tt.func @from_ptr_tensor(%arg0: tensor<4x!tt.ptr<f16>>) {
    %0 = tta.from_tt_ptr %arg0 : tensor<4x!tt.ptr<f16>> to !tta.addr<f16, 1, 1>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @from_block_ptr(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<tensor<2x3xi32>, 3>) {
// CHECK:           %[[FROM_TT_PTR_0:.*]] = tta.from_tt_ptr %[[ARG0]] : !tt.ptr<tensor<2x3xi32>, 3> to !tta.addr<i32, 2, 3>
// CHECK:           tt.return
// CHECK:         }
  tt.func @from_block_ptr(%arg0: !tt.ptr<tensor<2x3xi32>, 3>) {
    %0 = tta.from_tt_ptr %arg0 : !tt.ptr<tensor<2x3xi32>, 3> to !tta.addr<i32, 2, 3>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @from_ptr_tensor_addrspace(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4x!tt.ptr<f16, 5>>) {
// CHECK:           %[[FROM_TT_PTR_0:.*]] = tta.from_tt_ptr %[[ARG0]] : tensor<4x!tt.ptr<f16, 5>> to !tta.addr<f16, 1, 5>
// CHECK:           tt.return
// CHECK:         }
  tt.func @from_ptr_tensor_addrspace(%arg0: tensor<4x!tt.ptr<f16, 5>>) {
    %0 = tta.from_tt_ptr %arg0 : tensor<4x!tt.ptr<f16, 5>> to !tta.addr<f16, 1, 5>
    tt.return
  }
}
