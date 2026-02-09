// RUN: triton-xyz-opt --split-input-file --verify-diagnostics %s

module {
  tt.func @from_ptr_tensor_rank_mismatch(%arg0: tensor<4x!tt.ptr<f16>>) {
    // expected-error@+1 {{result rank must match imported pointer rank}}
    %0 = tta.from_tt_ptr %arg0 : tensor<4x!tt.ptr<f16>> to !tta.addr<f16, 2, 1>
    tt.return
  }
}

// -----

module {
  tt.func @from_block_ptr_space_mismatch(%arg0: !tt.ptr<tensor<2x3xi32>, 3>) {
    // expected-error@+1 {{result address space must match source address space}}
    %0 = tta.from_tt_ptr %arg0 : !tt.ptr<tensor<2x3xi32>, 3> to !tta.addr<i32, 2, 1>
    tt.return
  }
}

// -----

module {
  tt.func @from_block_ptr_elem_mismatch(%arg0: !tt.ptr<tensor<2x3xi32>, 3>) {
    // expected-error@+1 {{result element type must match source pointee element type}}
    %0 = tta.from_tt_ptr %arg0 : !tt.ptr<tensor<2x3xi32>, 3> to !tta.addr<f32, 2, 3>
    tt.return
  }
}

// -----

module {
  tt.func @from_invalid_result_not_addr(%arg0: !tt.ptr<f32>) {
    // expected-error@+1 {{op result #0 must be tta.addr type, but got '!tt.ptr<f32>'}}
    %0 = tta.from_tt_ptr %arg0 : !tt.ptr<f32> to !tt.ptr<f32>
    tt.return
  }
}
