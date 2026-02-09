// RUN: triton-xyz-opt --split-input-file %s | FileCheck %s

module {
  // CHECK-LABEL: tt.func @from_scalar_ptr(
  // CHECK-SAME: %[[ARG:.*]]: !tt.ptr<f32>
  // CHECK: %[[ADDR:.*]] = tta.from_tt_ptr %[[ARG]] : !tt.ptr<f32> to !tta.addr<f32, 1, 1>
  tt.func @from_scalar_ptr(%arg0: !tt.ptr<f32>) {
    %0 = tta.from_tt_ptr %arg0 : !tt.ptr<f32> to !tta.addr<f32, 1, 1>
    tt.return
  }
}

// -----

module {
  // CHECK-LABEL: tt.func @from_ptr_tensor(
  // CHECK-SAME: %[[ARG:.*]]: tensor<4x!tt.ptr<f16>>
  // CHECK: %[[ADDR:.*]] = tta.from_tt_ptr %[[ARG]] : tensor<4x!tt.ptr<f16>> to !tta.addr<f16, 1, 1>
  tt.func @from_ptr_tensor(%arg0: tensor<4x!tt.ptr<f16>>) {
    %0 = tta.from_tt_ptr %arg0 : tensor<4x!tt.ptr<f16>> to !tta.addr<f16, 1, 1>
    tt.return
  }
}

// -----

module {
  // CHECK-LABEL: tt.func @from_block_ptr(
  // CHECK-SAME: %[[ARG:.*]]: !tt.ptr<tensor<2x3xi32>, 3>
  // CHECK: %[[ADDR:.*]] = tta.from_tt_ptr %[[ARG]] : !tt.ptr<tensor<2x3xi32>, 3> to !tta.addr<i32, 2, 3>
  tt.func @from_block_ptr(%arg0: !tt.ptr<tensor<2x3xi32>, 3>) {
    %0 = tta.from_tt_ptr %arg0 : !tt.ptr<tensor<2x3xi32>, 3> to !tta.addr<i32, 2, 3>
    tt.return
  }
}

// -----

module {
  // CHECK-LABEL: tt.func @from_ptr_tensor_addrspace(
  // CHECK-SAME: %[[ARG:.*]]: tensor<4x!tt.ptr<f16, 5>>
  // CHECK: %[[ADDR:.*]] = tta.from_tt_ptr %[[ARG]] : tensor<4x!tt.ptr<f16, 5>> to !tta.addr<f16, 1, 5>
  tt.func @from_ptr_tensor_addrspace(%arg0: tensor<4x!tt.ptr<f16, 5>>) {
    %0 = tta.from_tt_ptr %arg0 : tensor<4x!tt.ptr<f16, 5>> to !tta.addr<f16, 1, 5>
    tt.return
  }
}
