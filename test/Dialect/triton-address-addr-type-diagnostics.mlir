// RUN: triton-xyz-opt --split-input-file --verify-diagnostics %s

module {
  // expected-error @+1 {{element type must be scalar (non-pointer, non-tensor)}}
  tt.func @invalid_addr_type_ptr_elem(%arg0: !tta.addr<!tt.ptr<f32>, 2, 1>) {
    tt.return
  }
}

// -----

module {
  // expected-error @+1 {{element type must be scalar (non-pointer, non-tensor)}}
  tt.func @invalid_addr_type_tensor_elem(%arg0: !tta.addr<tensor<2xf32>, 2, 1>) {
    tt.return
  }
}

// -----

module {
  // expected-error @+1 {{rank must be greater than 0}}
  tt.func @invalid_addr_type_zero_rank(%arg0: !tta.addr<f32, 0, 1>) {
    tt.return
  }
}

// -----

module {
  // expected-error @+1 {{rank must be greater than 0}}
  tt.func @invalid_addr_type_negative_rank(%arg0: !tta.addr<f32, -1, 1>) {
    tt.return
  }
}

// -----

module {
  // expected-error @+1 {{address space must be non-negative}}
  tt.func @invalid_addr_type_negative_space(%arg0: !tta.addr<f32, 2, -1>) {
    tt.return
  }
}
