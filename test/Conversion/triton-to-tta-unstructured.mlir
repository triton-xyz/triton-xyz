// RUN: triton-xyz-opt --split-input-file --triton-to-tta-unstructured %s | FileCheck %s

module {
// CHECK-LABEL: tt.func public @masked_gather_scatter(
// CHECK: %[[A0:.*]] = tta.make_addr %arg0 to sizes: [4]
// CHECK: %[[R0:.*]] = "tta.reindex"(%[[A0]], %{{.*}}, %{{.*}})
// CHECK: %[[L0:.*]] = "tta.load"(%[[R0]], %{{.*}})
// CHECK: %[[A1:.*]] = tta.make_addr %arg1 to sizes: [4]
// CHECK: %[[R1:.*]] = "tta.reindex"(%[[A1]], %{{.*}}, %{{.*}})
// CHECK: "tta.store"(%[[R1]], %[[L0]])
  tt.func public @masked_gather_scatter(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %limit = tt.splat %arg2 : i32 -> tensor<4xi32>
    %mask = arith.cmpi slt, %range, %limit : tensor<4xi32>
    %zero = arith.constant 0.0 : f32
    %other = tt.splat %zero : f32 -> tensor<4xf32>
    %val = tt.load %in_ptrs, %mask, %other : tensor<4x!tt.ptr<f32>>
    tt.store %out_ptrs, %val, %mask : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @bitcast_ptr_chain(
// CHECK: %[[BC:.*]] = tt.bitcast %arg0 : !tt.ptr<f32> -> !tt.ptr<i32>
// CHECK: %[[A0:.*]] = tta.make_addr %[[BC]] to sizes: [4]
// CHECK: %[[R0:.*]] = "tta.reindex"(%[[A0]], %{{.*}})
// CHECK: %[[L0:.*]] = "tta.load"(%[[R0]], %{{.*}})
// CHECK: tt.return %[[L0]] : tensor<4xi32>
  tt.func @bitcast_ptr_chain(%arg0: !tt.ptr<f32>) -> tensor<4xi32> {
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %bc = tt.bitcast %arg0 : !tt.ptr<f32> -> !tt.ptr<i32>
    %base = tt.splat %bc : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %ptrs = tt.addptr %base, %r : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %vals = tt.load %ptrs : tensor<4x!tt.ptr<i32>>
    tt.return %vals : tensor<4xi32>
  }
}

// -----

module {
// CHECK-LABEL: tt.func @int_to_ptr_root_lowering(
// CHECK: %[[P:.*]] = tt.int_to_ptr %arg0 : i64 -> !tt.ptr<f32>
// CHECK: %[[A0:.*]] = tta.make_addr %[[P]] to sizes: [4]
// CHECK: %[[R0:.*]] = "tta.reindex"(%[[A0]], %{{.*}})
// CHECK: %[[L0:.*]] = "tta.load"(%[[R0]], %{{.*}})
// CHECK: tt.return %[[L0]] : tensor<4xf32>
  tt.func @int_to_ptr_root_lowering(%arg0: i64) -> tensor<4xf32> {
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %p = tt.int_to_ptr %arg0 : i64 -> !tt.ptr<f32>
    %base = tt.splat %p : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %vals = tt.load %ptrs : tensor<4x!tt.ptr<f32>>
    tt.return %vals : tensor<4xf32>
  }
}

// -----

module {
// CHECK-LABEL: tt.func @scalar_atomic_rmw_to_tta(
// CHECK: %[[A0:.*]] = "tta.atomic"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{kind = "add"}> : (!tta.addr<i32, 1, 1>, i32, i32, i1) -> i32
  tt.func @scalar_atomic_rmw_to_tta(%arg0: !tt.ptr<i32>, %arg1: i32, %arg2: i1) -> i32 {
    %a0 = tt.atomic_rmw add, acq_rel, gpu, %arg0, %arg1, %arg2 : (!tt.ptr<i32>, i32, i1) -> i32
    tt.return %a0 : i32
  }
}

// -----

module {
// CHECK-LABEL: tt.func public @offset_width_upgrade(
// CHECK: arith.extsi %{{.*}} : tensor<4xi32> to tensor<4xi64>
// CHECK: %[[A0:.*]] = tta.make_addr %arg0 to sizes: [4]
// CHECK: %[[R0:.*]] = "tta.reindex"(%[[A0]], %{{.*}})
// CHECK: %[[L0:.*]] = "tta.load"(%[[R0]], %{{.*}})
// CHECK: %[[A1:.*]] = tta.make_addr %arg1 to sizes: [4]
// CHECK: "tta.store"(%{{.*}}, %[[L0]])
  tt.func public @offset_width_upgrade(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %c5_i64 = arith.constant 5 : i64
    %off64 = tt.splat %c5_i64 : i64 -> tensor<4xi64>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptr0 = tt.addptr %base, %off64 : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
    %ptr1 = tt.addptr %ptr0, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %val = tt.load %ptr1 : tensor<4x!tt.ptr<f32>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %out_ptrs, %val : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func public @loop_ptr_iter_args(
// CHECK: scf.for
// CHECK: iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (tensor<4xi32>, tensor<4xi32>)
// CHECK: %[[A0:.*]] = tta.make_addr %arg0 to sizes: [4]
// CHECK: %[[R0:.*]] = "tta.reindex"(%[[A0]], %{{.*}})
// CHECK: %[[V0:.*]] = "tta.load"(%[[R0]], %{{.*}})
// CHECK: %[[A1:.*]] = tta.make_addr %arg1 to sizes: [4]
// CHECK: %[[R1:.*]] = "tta.reindex"(%[[A1]], %{{.*}})
// CHECK: "tta.store"(%[[R1]], %[[V0]])
  tt.func public @loop_ptr_iter_args(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %out_ptrs = tt.addptr %out_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %res:2 = scf.for %iv = %c0_i32 to %arg2 step %c1_i32 iter_args(%in = %in_ptrs, %out = %out_ptrs) -> (tensor<4x!tt.ptr<f32>>, tensor<4x!tt.ptr<f32>>)  : i32 {
      %val = tt.load %in : tensor<4x!tt.ptr<f32>>
      tt.store %out, %val : tensor<4x!tt.ptr<f32>>
      %next_in = tt.addptr %in, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %next_out = tt.addptr %out, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %next_in, %next_out : tensor<4x!tt.ptr<f32>>, tensor<4x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func public @masked_2d_fallback(
// CHECK: tensor.collapse_shape %{{.*}} {{.*}} : tensor<2x4xi32> into tensor<8xi32>
// CHECK: %[[A0:.*]] = tta.make_addr %arg0 to sizes: [8]
// CHECK: %[[R0:.*]] = "tta.reindex"(%[[A0]], %{{.*}}, %{{.*}})
// CHECK: %[[L0:.*]] = "tta.load"(%[[R0]], %{{.*}})
// CHECK: tensor.expand_shape %[[L0]] {{.*}} output_shape [2, 4] : tensor<8xf32> into tensor<2x4xf32>
// CHECK: %[[A1:.*]] = tta.make_addr %arg1 to sizes: [8]
// CHECK: "tta.store"(%{{.*}}, %{{.*}})
  tt.func public @masked_2d_fallback(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %col = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %row_exp = tt.expand_dims %row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %col_exp = tt.expand_dims %col {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %row_bcast = tt.broadcast %row_exp : tensor<2x1xi32> -> tensor<2x4xi32>
    %col_bcast = tt.broadcast %col_exp : tensor<1x4xi32> -> tensor<2x4xi32>
    %c4 = arith.constant 4 : i32
    %stride = tt.splat %c4 : i32 -> tensor<2x4xi32>
    %row_linear = arith.muli %row_bcast, %stride : tensor<2x4xi32>
    %offsets = arith.addi %row_linear, %col_bcast : tensor<2x4xi32>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_base, %offsets : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    %limit = tt.splat %arg2 : i32 -> tensor<2x4xi32>
    %mask = arith.cmpi slt, %offsets, %limit : tensor<2x4xi32>
    %zero = arith.constant 0.0 : f32
    %other = tt.splat %zero : f32 -> tensor<2x4xf32>
    %val = tt.load %in_ptrs, %mask, %other : tensor<2x4x!tt.ptr<f32>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %offsets : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    tt.store %out_ptrs, %val, %mask : tensor<2x4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @scalar_atomic_rmw_and_cas_to_tta(
// CHECK: %[[A0:.*]] = "tta.atomic"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{kind = "add"}> : (!tta.addr<i32, 1, 1>, i32, i32, i1) -> i32
// CHECK: %[[A1:.*]] = "tta.atomic"(%{{.*}}, %{{.*}}, %{{.*}}) <{kind = "xchg"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
// CHECK: %[[A2:.*]] = "tta.atomic_cas"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!tta.addr<i32, 1, 1>, i32, i32, i32) -> i32
  tt.func @scalar_atomic_rmw_and_cas_to_tta(%arg0: !tt.ptr<i32>, %arg1: i32, %arg2: i32, %arg3: i1) -> i32 {
    %a0 = tt.atomic_rmw add, acq_rel, gpu, %arg0, %arg1, %arg3 : (!tt.ptr<i32>, i32, i1) -> i32
    %a1 = tt.atomic_rmw exch, acq_rel, gpu, %arg0, %arg2 : (!tt.ptr<i32>, i32) -> i32
    %a2 = tt.atomic_cas acq_rel, gpu, %arg0, %arg1, %arg2 : (!tt.ptr<i32>, i32, i32) -> i32
    %sum0 = arith.addi %a0, %a1 : i32
    %sum = arith.addi %sum0, %a2 : i32
    tt.return %sum : i32
  }
}

// -----

module {
// CHECK-LABEL: tt.func @ptr_to_int_scalar_materialized(
// CHECK: %[[A0:.*]] = tta.make_addr %arg0 to sizes: [4]
// CHECK: %[[R0:.*]] = "tta.reindex"(%[[A0]], %{{.*}})
// CHECK: %[[V0:.*]] = "tta.load"(%[[R0]], %{{.*}})
// CHECK: %[[A1:.*]] = tta.make_addr %arg1 to sizes: [4]
// CHECK: "tta.store"(%{{.*}}, %[[V0]])
// CHECK: %[[OFF:.*]] = arith.addi %{{.*}}, %arg2 : i32
// CHECK: %[[P:.*]] = tt.addptr %arg0, %[[OFF]] : !tt.ptr<f32>, i32
// CHECK: %[[I:.*]] = tt.ptr_to_int %[[P]] : !tt.ptr<f32> -> i64
// CHECK: tt.return %[[I]] : i64
  tt.func @ptr_to_int_scalar_materialized(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) -> i64 {
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %vals = tt.load %ptrs : tensor<4x!tt.ptr<f32>>
    %outbase = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %outptrs = tt.addptr %outbase, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %outptrs, %vals : tensor<4x!tt.ptr<f32>>
    %p = tt.addptr %arg0, %arg2 : !tt.ptr<f32>, i32
    %i = tt.ptr_to_int %p : !tt.ptr<f32> -> i64
    tt.return %i : i64
  }
}

// -----

module {
// CHECK-LABEL: tt.func @make_tensor_ptr_accumulate_offset(
// CHECK: %[[A0:.*]] = tta.make_addr %arg0 to sizes: [4]
// CHECK: %[[R0:.*]] = "tta.reindex"(%[[A0]], %{{.*}})
// CHECK: %[[V0:.*]] = "tta.load"(%[[R0]], %{{.*}})
// CHECK: %[[BASE_OFF:.*]] = arith.addi %{{.*}}, %arg2 : i32
// CHECK: %[[ACC_OFF:.*]] = arith.addi %[[BASE_OFF]], %{{.*}} : i32
// CHECK: tt.make_tensor_ptr %arg0{{.*}}[%[[ACC_OFF]], %{{.*}}] {order = array<i32: 1, 0>} : <tensor<4x4xf16>>
// CHECK: %[[A1:.*]] = tta.make_addr %arg1 to sizes: [4]
// CHECK: "tta.store"(%{{.*}}, %[[V0]])
  tt.func @make_tensor_ptr_accumulate_offset(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: i32) {
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>

    %seed = tt.splat %arg0 : !tt.ptr<f16> -> tensor<4x!tt.ptr<f16>>
    %seed_ptrs = tt.addptr %seed, %r : tensor<4x!tt.ptr<f16>>, tensor<4xi32>
    %seed_vals = tt.load %seed_ptrs : tensor<4x!tt.ptr<f16>>

    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %base2 = tt.addptr %arg0, %arg2 : !tt.ptr<f16>, i32
    %tptr = tt.make_tensor_ptr %base2, [%c4_i64, %c4_i64], [%c4_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<4x4xf16>>
    %loaded = tt.load %tptr : !tt.ptr<tensor<4x4xf16>>
    tt.store %tptr, %loaded : !tt.ptr<tensor<4x4xf16>>

    %outbase = tt.splat %arg1 : !tt.ptr<f16> -> tensor<4x!tt.ptr<f16>>
    %outptrs = tt.addptr %outbase, %r : tensor<4x!tt.ptr<f16>>, tensor<4xi32>
    tt.store %outptrs, %seed_vals : tensor<4x!tt.ptr<f16>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @select_same_base_ptr_offsets_to_tta(
// CHECK: %[[OFF0:.*]] = arith.addi
// CHECK: %[[OFF1:.*]] = arith.addi
// CHECK: %[[SEL:.*]] = arith.select %{{.*}}, %[[OFF0]], %[[OFF1]] : tensor<4xi1>, tensor<4xi32>
// CHECK: %[[A0:.*]] = tta.make_addr %arg0 to sizes: [4]
// CHECK: %[[R0:.*]] = "tta.reindex"(%[[A0]], %[[SEL]])
// CHECK: %[[L0:.*]] = "tta.load"(%[[R0]], %{{.*}})
// CHECK: tt.return %[[L0]] : tensor<4xi32>
  tt.func @select_same_base_ptr_offsets_to_tta(%arg0: !tt.ptr<i32>) -> tensor<4xi32> {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %base = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %ptr0 = tt.addptr %base, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %c1 = arith.constant dense<1> : tensor<4xi32>
    %ptr1 = tt.addptr %ptr0, %c1 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %c2 = arith.constant dense<2> : tensor<4xi32>
    %rem = arith.remsi %range, %c2 : tensor<4xi32>
    %c0 = arith.constant dense<0> : tensor<4xi32>
    %mask = arith.cmpi eq, %rem, %c0 : tensor<4xi32>
    %ptrs = arith.select %mask, %ptr0, %ptr1 : tensor<4xi1>, tensor<4x!tt.ptr<i32>>
    %vals = tt.load %ptrs : tensor<4x!tt.ptr<i32>>
    tt.return %vals : tensor<4xi32>
  }
}
