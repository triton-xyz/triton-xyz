// RUN: triton-xyz-opt --verify-tta-bridge-eliminated --split-input-file --verify-diagnostics %s

module {
  tt.func @bridge_leftover(%arg0: !tt.ptr<f32>) {
    // expected-error@+1 {{'tta.from_tt_ptr' op must be eliminated before tta mid-lowering stage}}
    %0 = tta.from_tt_ptr %arg0 : !tt.ptr<f32> to !tta.addr<f32, 1, 1>
    tt.return
  }
}

// -----

module {
  tt.func @bridge_free(%arg0: !tt.ptr<f32>) {
    %0 = tta.make_addr %arg0 to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %1 = "tta.reindex"(%0) <{static_offsets = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
    %2 = "tta.advance"(%1) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
    %3 = "tta.load"(%2) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
    "tta.store"(%2, %3) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
    tt.return
  }
}
