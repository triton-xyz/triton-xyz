// RUN: triton-xyz-opt --split-input-file %s | FileCheck %s

module {
  // CHECK-LABEL: tt.func @addr_type_roundtrip(
  // CHECK-SAME: %[[A0:.*]]: !tta.addr<f32, 1, 1>
  // CHECK-SAME: %[[A1:.*]]: !tta.addr<i32, 2, 3>
  // CHECK-SAME: %[[A2:.*]]: !tta.addr<bf16, 3, 0>
  // CHECK-SAME: -> (!tta.addr<f32, 1, 1>, !tta.addr<i32, 2, 3>, !tta.addr<bf16, 3, 0>)
  tt.func @addr_type_roundtrip(%a0: !tta.addr<f32, 1, 1>,
                               %a1: !tta.addr<i32, 2, 3>,
                               %a2: !tta.addr<bf16, 3, 0>)
      -> (!tta.addr<f32, 1, 1>, !tta.addr<i32, 2, 3>, !tta.addr<bf16, 3, 0>) {
    tt.return %a0, %a1, %a2 : !tta.addr<f32, 1, 1>, !tta.addr<i32, 2, 3>, !tta.addr<bf16, 3, 0>
  }
}
