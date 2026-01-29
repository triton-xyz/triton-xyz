// RUN: triton-xyz-opt --split-input-file --verify-diagnostics --triton-to-unstructured %s

module {
  tt.func public @unsupported_cat(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %p0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %p1 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %c0 = tt.addptr %p0, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %c1 = tt.addptr %p1, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %cat = tt.cat %c0, %c1 : tensor<4x!tt.ptr<f32>> -> tensor<8x!tt.ptr<f32>>
    %val = tt.load %cat : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module { // expected-warning {{Cannot transform tensor of pointers into a single base pointer with tensor of offsets}}
  tt.func public @addptr_in_if(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cond = arith.cmpi eq, %arg2, %c0_i32 : i32
    %ptr = scf.if %cond -> (!tt.ptr<f32>) {
      %p0 = tt.addptr %arg0, %arg2 : !tt.ptr<f32>, i32
      scf.yield %p0 : !tt.ptr<f32>
    } else {
      %p1 = tt.addptr %arg1, %arg2 : !tt.ptr<f32>, i32
      scf.yield %p1 : !tt.ptr<f32>
    }
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %base = tt.splat %ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %val = tt.load %ptrs : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module { // expected-warning {{Cannot transform tensor of pointers into a single base pointer with tensor of offsets}}
  tt.func public @masked_load_non_splat_other(%arg0: !tt.ptr<f32>, %arg1: i32) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %limit = tt.splat %arg1 : i32 -> tensor<4xi32>
    %mask = arith.cmpi slt, %range, %limit : tensor<4xi32>
    %c0 = arith.constant 0.0 : f32
    %c1 = arith.constant 1.0 : f32
    %s0 = tt.splat %c0 : f32 -> tensor<4xf32>
    %s1 = tt.splat %c1 : f32 -> tensor<4xf32>
    %other = arith.addf %s0, %s1 : tensor<4xf32>
    // expected-error@+2 {{other value used in masked load produced by unsupported instruction}}
    // expected-error@+1 {{cannot parse `other` value for load}}
    %val = tt.load %ptrs, %mask, %other : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}
