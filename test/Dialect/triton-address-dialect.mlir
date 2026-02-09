// RUN: triton-xyz-opt --split-input-file %s | FileCheck %s

module {
// CHECK-LABEL: tt.func @address_reindex_advance_addr_type(
  tt.func @address_reindex_advance_addr_type(%a: !tta.addr<f32, 2, 1>, %idx: tensor<2xi32>, %mask: tensor<2xi1>) {
    // CHECK: "tta.reindex"(%
    %r0 = "tta.reindex"(%a) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 1>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    // CHECK: "tta.reindex"(%
    %r1 = "tta.reindex"(%r0, %idx, %mask) <{indirect_dim = 1 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0, 0>}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>, tensor<2xi1>) -> !tta.addr<f32, 2, 1>
    // CHECK: "tta.advance"(%
    %r2 = "tta.advance"(%r1) <{static_deltas = array<i64: 1, 0>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @address_mem_ops_addr_type(
  tt.func @address_mem_ops_addr_type(%a: !tta.addr<f32, 2, 1>, %off: i32, %val: tensor<?x?xf32>) {
    %other = arith.constant 0.000000e+00 : f32
    // CHECK: "tta.load"(%
    %loaded = "tta.load"(%a, %other) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, f32) -> tensor<?x?xf32>
    // CHECK: "tta.store"(%
    "tta.store"(%a, %loaded) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<?x?xf32>) -> ()

    // CHECK: "tta.atomic"(%
    %a0 = "tta.atomic"(%a, %off, %other) <{kind = "add"}> : (!tta.addr<f32, 2, 1>, i32, f32) -> f32
    "tta.store"(%a, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<?x?xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @address_core_ops(
  tt.func @address_core_ops(%base: !tt.ptr<f32>, %off: i32, %ibase: !tt.ptr<i32>, %ioff: i32) {
    %idx = arith.constant dense<[0, 1]> : tensor<2xi32>
    %mask = arith.constant dense<[true, false]> : tensor<2xi1>
    %smask = arith.constant true
    %val = arith.constant 1.000000e+00 : f32
    %other = arith.constant 0.000000e+00 : f32
    %ival = arith.constant 1 : i32

    // CHECK: tta.make_addr
    %addr = tta.make_addr %base to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>

    // CHECK: "tta.reindex"(%
    %r0 = "tta.reindex"(%addr) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 1>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>

    // CHECK: "tta.reindex"(%
    %r1 = "tta.reindex"(%r0, %idx, %mask) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0, 0>}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>, tensor<2xi1>) -> !tta.addr<f32, 2, 1>

    // CHECK: "tta.advance"(%
    %r2 = "tta.advance"(%r1) <{static_deltas = array<i64: 1, 0>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>

    // CHECK: "tta.load"(%
    %loaded = "tta.load"(%r2) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>) -> tensor<2x2xf32>
    // CHECK: "tta.load"(%
    %loadedOther = "tta.load"(%r2, %other) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, f32) -> tensor<2x2xf32>
    // CHECK: "tta.store"(%
    "tta.store"(%r2, %loaded) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<2x2xf32>) -> ()

    %base_i = tta.from_tt_ptr %base : !tt.ptr<f32> to !tta.addr<f32, 1, 1>
    %ibase_i = tta.from_tt_ptr %ibase : !tt.ptr<i32> to !tta.addr<i32, 1, 1>
    // CHECK: "tta.atomic"(%
    %a0 = "tta.atomic"(%base_i, %off, %val) <{kind = "add"}> : (!tta.addr<f32, 1, 1>, i32, f32) -> f32
    // CHECK: "tta.atomic"(%
    %a1 = "tta.atomic"(%base_i, %off, %a0, %smask) <{kind = "fadd"}> : (!tta.addr<f32, 1, 1>, i32, f32, i1) -> f32
    // CHECK: "tta.atomic"(%
    %a2 = "tta.atomic"(%ibase_i, %ioff, %ival) <{kind = "xor"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32

    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @address_mixed_dynamic_make_addr(
  tt.func @address_mixed_dynamic_make_addr(%base: !tt.ptr<f32>, %s0: index, %o0: index) {
    // CHECK: tta.make_addr
    %addr = tta.make_addr %base to sizes: [2, 2], strides: [%s0, 1], offsets: [%o0, 0], shape: [2, %s0], order: [1, 0] : <f32> to !tta.addr<f32, 2, 1>
    // CHECK: "tta.reindex"(%
    %r = "tta.reindex"(%addr) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    // CHECK: "tta.advance"(%
    %a = "tta.advance"(%r) <{static_deltas = array<i64: 0, 1>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    // CHECK: "tta.load"(%
    %v = "tta.load"(%a) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>) -> tensor<2x2xf32>
    // CHECK: "tta.store"(%
    "tta.store"(%a, %v) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<2x2xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @address_scalar_ptr_reindex_advance(
  tt.func @address_scalar_ptr_reindex_advance(%base: !tt.ptr<f32>) {
    %off = arith.constant 0 : i32
    %val = arith.constant 2.000000e+00 : f32
    %base_i = tta.from_tt_ptr %base : !tt.ptr<f32> to !tta.addr<f32, 1, 1>
    // CHECK: "tta.reindex"(%
    %r0 = "tta.reindex"(%base_i) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
    // CHECK: "tta.advance"(%
    %r1 = "tta.advance"(%r0) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
    // CHECK: "tta.atomic"(%
    %a = "tta.atomic"(%r1, %off, %val) <{kind = "xchg"}> : (!tta.addr<f32, 1, 1>, i32, f32) -> f32
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @address_masked_load_store(
  tt.func @address_masked_load_store(%base: !tt.ptr<f32>) {
    %m = arith.constant 1 : index
    %other = arith.constant 0.000000e+00 : f32
    %addr = tta.make_addr %base to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
    // CHECK: "tta.load"(%{{.*}}, %{{.*}}, %{{.*}})
    %v = "tta.load"(%addr, %m, %other) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: -9223372036854775808, 2>}> : (!tta.addr<f32, 2, 1>, index, f32) -> tensor<2x2xf32>
    // CHECK: "tta.store"(%{{.*}}, %{{.*}}, %{{.*}})
    "tta.store"(%addr, %v, %m) <{static_mask_dims = array<i64: -9223372036854775808, 2>}> : (!tta.addr<f32, 2, 1>, tensor<2x2xf32>, index) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @address_atomic_all_kinds(
  tt.func @address_atomic_all_kinds(%ibase: !tt.ptr<i32>, %fbase: !tt.ptr<f32>, %off: i32) {
    %i = arith.constant 1 : i32
    %f = arith.constant 1.000000e+00 : f32
    %ibase_i = tta.from_tt_ptr %ibase : !tt.ptr<i32> to !tta.addr<i32, 1, 1>
    %fbase_i = tta.from_tt_ptr %fbase : !tt.ptr<f32> to !tta.addr<f32, 1, 1>
    // CHECK: "tta.atomic"(%{{.*}}, %{{.*}}, %{{.*}}) <{kind = "add"}>
    %r0 = "tta.atomic"(%ibase_i, %off, %i) <{kind = "add"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
    // CHECK: "tta.atomic"(%{{.*}}, %{{.*}}, %{{.*}}) <{kind = "and"}>
    %r1 = "tta.atomic"(%ibase_i, %off, %r0) <{kind = "and"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
    // CHECK: "tta.atomic"(%{{.*}}, %{{.*}}, %{{.*}}) <{kind = "or"}>
    %r2 = "tta.atomic"(%ibase_i, %off, %r1) <{kind = "or"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
    // CHECK: "tta.atomic"(%{{.*}}, %{{.*}}, %{{.*}}) <{kind = "xor"}>
    %r3 = "tta.atomic"(%ibase_i, %off, %r2) <{kind = "xor"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
    // CHECK: "tta.atomic"(%{{.*}}, %{{.*}}, %{{.*}}) <{kind = "max"}>
    %r4 = "tta.atomic"(%ibase_i, %off, %r3) <{kind = "max"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
    // CHECK: "tta.atomic"(%{{.*}}, %{{.*}}, %{{.*}}) <{kind = "min"}>
    %r5 = "tta.atomic"(%ibase_i, %off, %r4) <{kind = "min"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
    // CHECK: "tta.atomic"(%{{.*}}, %{{.*}}, %{{.*}}) <{kind = "xchg"}>
    %r6 = "tta.atomic"(%ibase_i, %off, %r5) <{kind = "xchg"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
    // CHECK: "tta.atomic"(%{{.*}}, %{{.*}}, %{{.*}}) <{kind = "cmpxchg"}>
    %r7 = "tta.atomic"(%ibase_i, %off, %r6) <{kind = "cmpxchg"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
    // CHECK: "tta.atomic"(%{{.*}}, %{{.*}}, %{{.*}}) <{kind = "fadd"}>
    %rf = "tta.atomic"(%fbase_i, %off, %f) <{kind = "fadd"}> : (!tta.addr<f32, 1, 1>, i32, f32) -> f32
    tt.return
  }
}
