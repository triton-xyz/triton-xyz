// RUN: triton-xyz-opt --split-input-file --verify-diagnostics %s

module {
  tt.func public @nop_non_tensor(%arg0: f32) -> f32 {
    // expected-error @+1 {{invalid kind of type specified: expected builtin.tensor, but found 'f32'}}
    %0 = tt.nop %arg0 : f32
    tt.return %0 : f32
  }
}
