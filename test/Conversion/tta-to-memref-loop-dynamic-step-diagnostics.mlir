// RUN: triton-xyz-opt --split-input-file --verify-diagnostics --tta-to-memref %s

module {
  tt.func @loop_carried_addr_unsupported_non_positive_constant_step(%src: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %step = arith.constant 0 : i32
    %addr0 = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], wrap_boundaries: [0], layout: "strided" : <f32> to !tta.addr<f32, 1, 1>
    %res = scf.for %iv = %c0 to %n step %step iter_args(%addr = %addr0) -> (!tta.addr<f32, 1, 1>) : i32 {
      // expected-error@+1 {{'tta.load' op unsupported loop-carried !tta.addr recurrence in scf.for iter_args}}
      %v = "tta.load"(%addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
      %next = "tta.advance"(%addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      scf.yield %next : !tta.addr<f32, 1, 1>
    }
    tt.return
  }
}
