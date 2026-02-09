// RUN: triton-xyz-opt --split-input-file --verify-diagnostics --tta-to-memref %s

module {
  tt.func @non_zero_indirect_offset(%src: !tt.ptr<f32>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %idx = "tta.reindex"(%addr, %offsets) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 1>}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
    // expected-error@+1 {{failed to legalize operation 'tta.load' that was explicitly marked illegal}}
    %val = "tta.load"(%idx) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
    tt.return
  }
}

// -----

module {
  tt.func @indirect_on_block_ptr(%base: !tt.ptr<f16>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %addr = tta.make_addr %base to sizes: [4, 4], strides: [4, 1], offsets: [0, 0], shape: [4, 4], order: [1, 0] : <f16> to !tta.addr<f16, 2, 1>
    %idx = "tta.reindex"(%addr, %offsets) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, 0>}> : (!tta.addr<f16, 2, 1>, tensor<4xi32>) -> !tta.addr<f16, 2, 1>
    // expected-error@+1 {{failed to legalize operation 'tta.load' that was explicitly marked illegal}}
    %val = "tta.load"(%idx) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f16, 2, 1>) -> tensor<4x4xf16>
    tt.return
  }
}

// -----

module {
  tt.func @loop_carried_addr_unsupported(%src: !tt.ptr<f32>, %n: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %addr0 = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %res = scf.for %iv = %c0 to %n step %step iter_args(%addr = %addr0) -> (!tta.addr<f32, 1, 1>) : i32 {
      // expected-error@+1 {{'tta.load' op unsupported loop-carried !tta.addr recurrence in scf.for iter_args}}
      %v = "tta.load"(%addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
      %next = "tta.advance"(%addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      scf.yield %next : !tta.addr<f32, 1, 1>
    }
    tt.return
  }
}


// -----

module {
  tt.func @loop_carried_addr_unsupported_seed(%seed: !tta.addr<f32, 1, 1>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %res = scf.for %iv = %c0 to %n step %c1 iter_args(%addr = %seed) -> (!tta.addr<f32, 1, 1>) : i32 {
      // expected-error@+1 {{'tta.load' op unsupported loop-carried !tta.addr recurrence in scf.for iter_args}}
      %v = "tta.load"(%addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
      %next = "tta.advance"(%addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      scf.yield %next : !tta.addr<f32, 1, 1>
    }
    tt.return
  }
}

// -----

module {
  tt.func @loop_carried_addr_unsupported_indirect_recurrence(%src: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %indices = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %addr0 = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %res = scf.for %iv = %c0 to %n step %c1 iter_args(%addr = %addr0) -> (!tta.addr<f32, 1, 1>) : i32 {
      // expected-error@+1 {{'tta.load' op unsupported loop-carried !tta.addr recurrence in scf.for iter_args}}
      %v = "tta.load"(%addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
      %next = "tta.reindex"(%addr, %indices) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
      scf.yield %next : !tta.addr<f32, 1, 1>
    }
    tt.return
  }
}

// -----

module {
  tt.func @loop_carried_addr_unsupported_dynamic_lower_bound(%src: !tt.ptr<f32>, %lb: i32, %n: i32) {
    %c1 = arith.constant 1 : i32
    %addr0 = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %res = scf.for %iv = %lb to %n step %c1 iter_args(%addr = %addr0) -> (!tta.addr<f32, 1, 1>) : i32 {
      // expected-error@+1 {{'tta.load' op unsupported loop-carried !tta.addr recurrence in scf.for iter_args}}
      %v = "tta.load"(%addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
      %next = "tta.advance"(%addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      scf.yield %next : !tta.addr<f32, 1, 1>
    }
    tt.return
  }
}

// -----

module {
  tt.func @loop_carried_addr_unsupported_dynamic_step(%src: !tt.ptr<f32>, %n: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %addr0 = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %res = scf.for %iv = %c0 to %n step %step iter_args(%addr = %addr0) -> (!tta.addr<f32, 1, 1>) : i32 {
      // expected-error@+1 {{'tta.load' op unsupported loop-carried !tta.addr recurrence in scf.for iter_args}}
      %v = "tta.load"(%addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
      %next = "tta.advance"(%addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      scf.yield %next : !tta.addr<f32, 1, 1>
    }
    tt.return
  }
}
