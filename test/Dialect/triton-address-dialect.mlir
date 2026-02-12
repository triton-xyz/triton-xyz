// RUN: triton-xyz-opt --split-input-file %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @address_reindex_advance_addr_type(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tta.addr<f32, 2, 1>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<2xi32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<2xi1>) {
// CHECK:           %[[VAL_0:.*]] = "tta.reindex"(%[[ARG0]]) <{static_offsets = array<i64: 0, 1>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
// CHECK:           %[[VAL_1:.*]] = "tta.indirect_reindex"(%[[VAL_0]], %[[ARG1]], %[[ARG2]]) <{indirect_dim = 1 : i32}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>, tensor<2xi1>) -> !tta.addr<f32, 2, 1>
// CHECK:           %[[VAL_2:.*]] = "tta.advance"(%[[VAL_1]]) <{static_deltas = array<i64: 1, 0>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
// CHECK:           tt.return
// CHECK:         }
  tt.func @address_reindex_advance_addr_type(%a: !tta.addr<f32, 2, 1>, %idx: tensor<2xi32>, %mask: tensor<2xi1>) {
    %r0 = "tta.reindex"(%a) <{static_offsets = array<i64: 0, 1>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    %r1 = "tta.indirect_reindex"(%r0, %idx, %mask) <{indirect_dim = 1 : i32}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>, tensor<2xi1>) -> !tta.addr<f32, 2, 1>
    %r2 = "tta.advance"(%r1) <{static_deltas = array<i64: 1, 0>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @address_mem_ops_addr_type(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tta.addr<f32, 2, 1>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<?x?xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_0:.*]] = "tta.load"(%[[ARG0]], %[[CONSTANT_0]]) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, f32) -> tensor<?x?xf32>
// CHECK:           "tta.store"(%[[ARG0]], %[[VAL_0]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<?x?xf32>) -> ()
// CHECK:           %[[VAL_1:.*]] = "tta.atomic"(%[[ARG0]], %[[ARG1]], %[[CONSTANT_0]]) <{kind = "add"}> : (!tta.addr<f32, 2, 1>, i32, f32) -> f32
// CHECK:           "tta.store"(%[[ARG0]], %[[ARG2]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<?x?xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @address_mem_ops_addr_type(%a: !tta.addr<f32, 2, 1>, %off: i32, %val: tensor<?x?xf32>) {
    %other = arith.constant 0.000000e+00 : f32
    %loaded = "tta.load"(%a, %other) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, f32) -> tensor<?x?xf32>
    "tta.store"(%a, %loaded) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<?x?xf32>) -> ()

    %a0 = "tta.atomic"(%a, %off, %other) <{kind = "add"}> : (!tta.addr<f32, 2, 1>, i32, f32) -> f32
    "tta.store"(%a, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<?x?xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @address_core_ops(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<[0, 1]> : tensor<2xi32>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant dense<[true, false]> : tensor<2xi1>
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant true
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 1 : i32
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.reindex"(%[[MAKE_ADDR_0]]) <{static_offsets = array<i64: 0, 1>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
// CHECK:           %[[VAL_1:.*]] = "tta.indirect_reindex"(%[[VAL_0]], %[[CONSTANT_0]], %[[CONSTANT_1]]) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>, tensor<2xi1>) -> !tta.addr<f32, 2, 1>
// CHECK:           %[[VAL_2:.*]] = "tta.advance"(%[[VAL_1]]) <{static_deltas = array<i64: 1, 0>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
// CHECK:           %[[VAL_3:.*]] = "tta.load"(%[[VAL_2]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>) -> tensor<2x2xf32>
// CHECK:           %[[VAL_4:.*]] = "tta.load"(%[[VAL_2]], %[[CONSTANT_4]]) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, f32) -> tensor<2x2xf32>
// CHECK:           "tta.store"(%[[VAL_2]], %[[VAL_3]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<2x2xf32>) -> ()
// CHECK:           %[[FROM_TT_PTR_0:.*]] = tta.from_tt_ptr %[[ARG0]] : !tt.ptr<f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[FROM_TT_PTR_1:.*]] = tta.from_tt_ptr %[[ARG2]] : !tt.ptr<i32> to !tta.addr<i32, 1, 1>
// CHECK:           %[[VAL_5:.*]] = "tta.atomic"(%[[FROM_TT_PTR_0]], %[[ARG1]], %[[CONSTANT_3]]) <{kind = "add"}> : (!tta.addr<f32, 1, 1>, i32, f32) -> f32
// CHECK:           %[[VAL_6:.*]] = "tta.atomic"(%[[FROM_TT_PTR_0]], %[[ARG1]], %[[VAL_5]], %[[CONSTANT_2]]) <{kind = "fadd"}> : (!tta.addr<f32, 1, 1>, i32, f32, i1) -> f32
// CHECK:           %[[VAL_7:.*]] = "tta.atomic"(%[[FROM_TT_PTR_1]], %[[ARG3]], %[[CONSTANT_5]]) <{kind = "xor"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
// CHECK:           tt.return
// CHECK:         }
  tt.func @address_core_ops(%base: !tt.ptr<f32>, %off: i32, %ibase: !tt.ptr<i32>, %ioff: i32) {
    %idx = arith.constant dense<[0, 1]> : tensor<2xi32>
    %mask = arith.constant dense<[true, false]> : tensor<2xi1>
    %smask = arith.constant true
    %val = arith.constant 1.000000e+00 : f32
    %other = arith.constant 0.000000e+00 : f32
    %ival = arith.constant 1 : i32

    %addr = tta.make_addr %base to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>

    %r0 = "tta.reindex"(%addr) <{static_offsets = array<i64: 0, 1>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>

    %r1 = "tta.indirect_reindex"(%r0, %idx, %mask) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>, tensor<2xi1>) -> !tta.addr<f32, 2, 1>

    %r2 = "tta.advance"(%r1) <{static_deltas = array<i64: 1, 0>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>

    %loaded = "tta.load"(%r2) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>) -> tensor<2x2xf32>
    %loadedOther = "tta.load"(%r2, %other) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, f32) -> tensor<2x2xf32>
    "tta.store"(%r2, %loaded) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<2x2xf32>) -> ()

    %base_i = tta.from_tt_ptr %base : !tt.ptr<f32> to !tta.addr<f32, 1, 1>
    %ibase_i = tta.from_tt_ptr %ibase : !tt.ptr<i32> to !tta.addr<i32, 1, 1>
    %a0 = "tta.atomic"(%base_i, %off, %val) <{kind = "add"}> : (!tta.addr<f32, 1, 1>, i32, f32) -> f32
    %a1 = "tta.atomic"(%base_i, %off, %a0, %smask) <{kind = "fadd"}> : (!tta.addr<f32, 1, 1>, i32, f32, i1) -> f32
    %a2 = "tta.atomic"(%ibase_i, %ioff, %ival) <{kind = "xor"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32

    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @address_mixed_dynamic_make_addr(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: index,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: index) {
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [2, 2], strides: {{\[}}%[[ARG1]], 1], offsets: {{\[}}%[[ARG2]], 0], shape: [2, %[[ARG1]]], order: [1, 0] : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.reindex"(%[[MAKE_ADDR_0]]) <{static_offsets = array<i64: 0, 0>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
// CHECK:           %[[VAL_1:.*]] = "tta.advance"(%[[VAL_0]]) <{static_deltas = array<i64: 0, 1>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
// CHECK:           %[[VAL_2:.*]] = "tta.load"(%[[VAL_1]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>) -> tensor<2x2xf32>
// CHECK:           "tta.store"(%[[VAL_1]], %[[VAL_2]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<2x2xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @address_mixed_dynamic_make_addr(%base: !tt.ptr<f32>, %s0: index, %o0: index) {
    %addr = tta.make_addr %base to sizes: [2, 2], strides: [%s0, 1], offsets: [%o0, 0], shape: [2, %s0], order: [1, 0] : <f32> to !tta.addr<f32, 2, 1>
    %r = "tta.reindex"(%addr) <{static_offsets = array<i64: 0, 0>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    %a = "tta.advance"(%r) <{static_deltas = array<i64: 0, 1>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    %v = "tta.load"(%a) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>) -> tensor<2x2xf32>
    "tta.store"(%a, %v) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<2x2xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @address_scalar_ptr_reindex_advance(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:           %[[FROM_TT_PTR_0:.*]] = tta.from_tt_ptr %[[ARG0]] : !tt.ptr<f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.reindex"(%[[FROM_TT_PTR_0]]) <{static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_1:.*]] = "tta.advance"(%[[VAL_0]]) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_2:.*]] = "tta.atomic"(%[[VAL_1]], %[[CONSTANT_0]], %[[CONSTANT_1]]) <{kind = "xchg"}> : (!tta.addr<f32, 1, 1>, i32, f32) -> f32
// CHECK:           tt.return
// CHECK:         }
  tt.func @address_scalar_ptr_reindex_advance(%base: !tt.ptr<f32>) {
    %off = arith.constant 0 : i32
    %val = arith.constant 2.000000e+00 : f32
    %base_i = tta.from_tt_ptr %base : !tt.ptr<f32> to !tta.addr<f32, 1, 1>
    %r0 = "tta.reindex"(%base_i) <{static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
    %r1 = "tta.advance"(%r0) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
    %a = "tta.atomic"(%r1, %off, %val) <{kind = "xchg"}> : (!tta.addr<f32, 1, 1>, i32, f32) -> f32
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @address_masked_load_store(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.load"(%[[MAKE_ADDR_0]], %[[CONSTANT_0]], %[[CONSTANT_1]]) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: -9223372036854775808, 2>}> : (!tta.addr<f32, 2, 1>, index, f32) -> tensor<2x2xf32>
// CHECK:           "tta.store"(%[[MAKE_ADDR_0]], %[[VAL_0]], %[[CONSTANT_0]]) <{static_mask_dims = array<i64: -9223372036854775808, 2>}> : (!tta.addr<f32, 2, 1>, tensor<2x2xf32>, index) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @address_masked_load_store(%base: !tt.ptr<f32>) {
    %m = arith.constant 1 : index
    %other = arith.constant 0.000000e+00 : f32
    %addr = tta.make_addr %base to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
    %v = "tta.load"(%addr, %m, %other) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: -9223372036854775808, 2>}> : (!tta.addr<f32, 2, 1>, index, f32) -> tensor<2x2xf32>
    "tta.store"(%addr, %v, %m) <{static_mask_dims = array<i64: -9223372036854775808, 2>}> : (!tta.addr<f32, 2, 1>, tensor<2x2xf32>, index) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @address_atomic_all_kinds(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[FROM_TT_PTR_0:.*]] = tta.from_tt_ptr %[[ARG0]] : !tt.ptr<i32> to !tta.addr<i32, 1, 1>
// CHECK:           %[[FROM_TT_PTR_1:.*]] = tta.from_tt_ptr %[[ARG1]] : !tt.ptr<f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.atomic"(%[[FROM_TT_PTR_0]], %[[ARG2]], %[[CONSTANT_0]]) <{kind = "add"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
// CHECK:           %[[VAL_1:.*]] = "tta.atomic"(%[[FROM_TT_PTR_0]], %[[ARG2]], %[[VAL_0]]) <{kind = "and"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
// CHECK:           %[[VAL_2:.*]] = "tta.atomic"(%[[FROM_TT_PTR_0]], %[[ARG2]], %[[VAL_1]]) <{kind = "or"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
// CHECK:           %[[VAL_3:.*]] = "tta.atomic"(%[[FROM_TT_PTR_0]], %[[ARG2]], %[[VAL_2]]) <{kind = "xor"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
// CHECK:           %[[VAL_4:.*]] = "tta.atomic"(%[[FROM_TT_PTR_0]], %[[ARG2]], %[[VAL_3]]) <{kind = "max"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
// CHECK:           %[[VAL_5:.*]] = "tta.atomic"(%[[FROM_TT_PTR_0]], %[[ARG2]], %[[VAL_4]]) <{kind = "min"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
// CHECK:           %[[VAL_6:.*]] = "tta.atomic"(%[[FROM_TT_PTR_0]], %[[ARG2]], %[[VAL_5]]) <{kind = "xchg"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
// CHECK:           %[[VAL_7:.*]] = "tta.atomic_cas"(%[[FROM_TT_PTR_0]], %[[ARG2]], %[[VAL_6]], %[[CONSTANT_0]]) : (!tta.addr<i32, 1, 1>, i32, i32, i32) -> i32
// CHECK:           %[[VAL_8:.*]] = "tta.atomic"(%[[FROM_TT_PTR_1]], %[[ARG2]], %[[CONSTANT_1]]) <{kind = "fadd"}> : (!tta.addr<f32, 1, 1>, i32, f32) -> f32
// CHECK:           tt.return
// CHECK:         }
  tt.func @address_atomic_all_kinds(%ibase: !tt.ptr<i32>, %fbase: !tt.ptr<f32>, %off: i32) {
    %i = arith.constant 1 : i32
    %f = arith.constant 1.000000e+00 : f32
    %ibase_i = tta.from_tt_ptr %ibase : !tt.ptr<i32> to !tta.addr<i32, 1, 1>
    %fbase_i = tta.from_tt_ptr %fbase : !tt.ptr<f32> to !tta.addr<f32, 1, 1>
    %r0 = "tta.atomic"(%ibase_i, %off, %i) <{kind = "add"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
    %r1 = "tta.atomic"(%ibase_i, %off, %r0) <{kind = "and"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
    %r2 = "tta.atomic"(%ibase_i, %off, %r1) <{kind = "or"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
    %r3 = "tta.atomic"(%ibase_i, %off, %r2) <{kind = "xor"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
    %r4 = "tta.atomic"(%ibase_i, %off, %r3) <{kind = "max"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
    %r5 = "tta.atomic"(%ibase_i, %off, %r4) <{kind = "min"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
    %r6 = "tta.atomic"(%ibase_i, %off, %r5) <{kind = "xchg"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
    %r7 = "tta.atomic_cas"(%ibase_i, %off, %r6, %i) : (!tta.addr<i32, 1, 1>, i32, i32, i32) -> i32
    %rf = "tta.atomic"(%fbase_i, %off, %f) <{kind = "fadd"}> : (!tta.addr<f32, 1, 1>, i32, f32) -> f32
    tt.return
  }
}
