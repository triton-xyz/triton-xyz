// RUN: triton-xyz-opt --split-input-file --verify-diagnostics --verify-tta-lowered %s

module {
  tt.func @leftover_tta_op(%arg0: !tt.ptr<f32>) {
    // expected-error@+1 {{'tta.from_tt_ptr' op must be eliminated before backend-ready stage}}
    %0 = tta.from_tt_ptr %arg0 : !tt.ptr<f32> to !tta.addr<f32, 1, 1>
    tt.return
  }
}

// -----

module {
  // expected-error@+1 {{'tt.func' op function signature contains !tta.addr after tta lowering}}
  tt.func @leftover_signature(%arg0: !tta.addr<f32, 1, 1>) {
    tt.return
  }
}

// -----

module {
  tt.func @leftover_tuple_result_type(%arg0: i32) {
    // expected-error@+1 {{result type contains !tta.addr after tta lowering}}
    %res = builtin.unrealized_conversion_cast %arg0 : i32 to tuple<!tta.addr<f32, 1, 1>>
    tt.return
  }
}
