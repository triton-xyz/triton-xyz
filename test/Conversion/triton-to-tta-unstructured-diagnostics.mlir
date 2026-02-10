// RUN: triton-xyz-opt --split-input-file --triton-to-tta-unstructured --remove-dead-values --canonicalize %s | FileCheck %s

module {
// CHECK-LABEL: tt.func @local_failure_cat_does_not_block(
// CHECK: %[[A0:.*]] = tta.make_addr %arg0 to sizes: [4]
// CHECK: %[[R0:.*]] = "tta.reindex"(%[[A0]],
// CHECK: %[[V0:.*]] = "tta.load"(%[[R0]],
// CHECK: tt.cat %{{.*}}, %{{.*}} {tta.fallback, tta.fallback_reason = "multi_base_cat_unsupported"}
// CHECK: tt.load %{{.*}} : tensor<4x!tt.ptr<f32>>
// CHECK: %[[A1:.*]] = tta.make_addr %arg2 to sizes: [4]
// CHECK: %[[R1:.*]] = "tta.reindex"(%[[A1]],
// CHECK: "tta.store"(%[[R1]],
  tt.func @local_failure_cat_does_not_block(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %r4 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>

    %good_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %good_ptrs = tt.addptr %good_base, %r4 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %good = tt.load %good_ptrs : tensor<4x!tt.ptr<f32>>

    %b0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
    %b1 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
    %r2 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %c0 = tt.addptr %b0, %r2 : tensor<2x!tt.ptr<f32>>, tensor<2xi32>
    %c1 = tt.addptr %b1, %r2 : tensor<2x!tt.ptr<f32>>, tensor<2xi32>
    %cat = tt.cat %c0, %c1 : tensor<2x!tt.ptr<f32>> -> tensor<4x!tt.ptr<f32>>
    %bad = tt.load %cat : tensor<4x!tt.ptr<f32>>

    %sum = arith.addf %good, %bad : tensor<4xf32>

    %out_base = tt.splat %arg2 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %r4 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %out_ptrs, %sum : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @fallback_other_not_scalar(
// CHECK: tt.load %{{.*}}, %{{.*}}, %{{.*}} {tta.fallback, tta.fallback_reason = "other_not_scalar_splat"}
// CHECK: %[[A0:.*]] = tta.make_addr %arg1 to sizes: [4]
// CHECK: %[[R0:.*]] = "tta.reindex"(%[[A0]], %{{.*}}, %{{.*}})
// CHECK: "tta.store"(%[[R0]],
  tt.func @fallback_other_not_scalar(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>

    %limit = tt.splat %arg2 : i32 -> tensor<4xi32>
    %mask = arith.cmpi slt, %range, %limit : tensor<4xi32>
    %c0 = arith.constant 0.0 : f32
    %c1 = arith.constant 1.0 : f32
    %s0 = tt.splat %c0 : f32 -> tensor<4xf32>
    %s1 = tt.splat %c1 : f32 -> tensor<4xf32>
    %other = arith.addf %s0, %s1 : tensor<4xf32>

    %v = tt.load %in_ptrs, %mask, %other : tensor<4x!tt.ptr<f32>>

    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %out_ptrs, %v, %mask : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @fallback_addptr_in_if_reason(
// CHECK: "tta.load"(%{{.*}}, %{{.*}})
// CHECK: tt.addptr %arg0, %{{.*}} {tta.fallback, tta.fallback_reason = "addptr_in_scf_if_unsupported"}
// CHECK: tt.load %{{.*}} : tensor<4x!tt.ptr<f32>>
// CHECK: "tta.store"(%{{.*}}, %{{.*}})
  tt.func @fallback_addptr_in_if_reason(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i1) {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>

    %good_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %good_ptrs = tt.addptr %good_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %good = tt.load %good_ptrs : tensor<4x!tt.ptr<f32>>

    %sel_ptr = scf.if %arg3 -> (!tt.ptr<f32>) {
      %a = tt.addptr %arg0, %c1_i32 : !tt.ptr<f32>, i32
      scf.yield %a : !tt.ptr<f32>
    } else {
      %b = tt.addptr %arg1, %c2_i32 : !tt.ptr<f32>, i32
      scf.yield %b : !tt.ptr<f32>
    }
    %bad_base = tt.splat %sel_ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %bad_ptrs = tt.addptr %bad_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %bad = tt.load %bad_ptrs : tensor<4x!tt.ptr<f32>>

    %sum = arith.addf %good, %bad : tensor<4xf32>
    %out_base = tt.splat %arg2 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %out_ptrs, %sum : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @fallback_atomic_kind_unsupported(
// CHECK: tt.atomic_rmw umax, acq_rel, gpu, %{{.*}}, %{{.*}}, %{{.*}} {tta.fallback, tta.fallback_reason = "atomic_kind_unsupported"}
  tt.func @fallback_atomic_kind_unsupported(%arg0: !tt.ptr<i32>, %arg1: i32, %arg2: i1) {
    %r = tt.atomic_rmw umax, acq_rel, gpu, %arg0, %arg1, %arg2 : (!tt.ptr<i32>, i32, i1) -> i32
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @overwrite_existing_fallback_reason(
// CHECK: tt.atomic_rmw umax, acq_rel, gpu, %{{.*}}, %{{.*}}, %{{.*}} {tta.fallback, tta.fallback_reason = "atomic_kind_unsupported"}
  tt.func @overwrite_existing_fallback_reason(%arg0: !tt.ptr<i32>, %arg1: i32, %arg2: i1) {
    %r = tt.atomic_rmw umax, acq_rel, gpu, %arg0, %arg1, %arg2 {tta.fallback, tta.fallback_reason = "pre_marked_reason"} : (!tt.ptr<i32>, i32, i1) -> i32
    tt.return
  }
}
