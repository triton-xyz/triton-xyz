// RUN: triton-shared-opt --split-input-file --verify-diagnostics --triton-to-structured="skip-prepass=true" %s

module {
  tt.func public @unsupported_mod_add(%arg0: !tt.ptr<f32>) {
    %range = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %c4 = arith.constant 4 : i32
    %mod = tt.splat %c4 : i32 -> tensor<8xi32>
    %rem = arith.remsi %range, %mod : tensor<8xi32>
    %offs = arith.addi %rem, %range : tensor<8xi32>
    %val, %off0, %stride0 = "tts.get_structured_state"(%offs) <{resultSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<8xi32>) -> (tensor<8xi32>, index, index) // expected-warning {{Rewriting GetStructuredStateOp failed.}}
    tt.return
  }
}

// -----

module {
  tt.func public @unsupported_block_arg(%arg0: tensor<4xi32>) {
    %val, %off0, %stride0 = "tts.get_structured_state"(%arg0) <{resultSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<4xi32>) -> (tensor<4xi32>, index, index) // expected-warning {{Rewriting GetStructuredStateOp failed.}}
    tt.return
  }
}
