// RUN: triton-xyz-opt --canonicalize --split-input-file %s | FileCheck %s

module {
  // CHECK-LABEL: tt.func @reindex_zero_fold(
  // CHECK: %[[BASE:.*]] = tta.make_addr %arg0
  // CHECK-NOT: "tta.reindex"
  // CHECK: tt.return %{{.*}}
  tt.func @reindex_zero_fold(%arg0: !tt.ptr<f32>) -> !tta.addr<f32, 2, 1> {
    %base = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
    %r = "tta.reindex"(%base) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    tt.return %r : !tta.addr<f32, 2, 1>
  }
}

// -----

module {
  // CHECK-LABEL: tt.func @advance_of_reindex_compose(
  // CHECK: %[[BASE:.*]] = tta.make_addr %arg0
  // CHECK: %[[R:.*]] = "tta.reindex"(%{{.*}}) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 8, 11>}>
  // CHECK-NOT: "tta.advance"
  // CHECK: tt.return %[[R]]
  tt.func @advance_of_reindex_compose(%arg0: !tt.ptr<f32>) -> !tta.addr<f32, 2, 1> {
    %base = tta.make_addr %arg0 to sizes: [16, 16], strides: [16, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
    %r0 = "tta.reindex"(%base) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 3, 4>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    %a1 = "tta.advance"(%r0) <{static_deltas = array<i64: 5, 7>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    tt.return %a1 : !tta.addr<f32, 2, 1>
  }
}

// -----

module {
  // CHECK-LABEL: tt.func @reindex_of_advance_compose(
  // CHECK: %[[BASE:.*]] = tta.make_addr %arg0
  // CHECK: %[[R:.*]] = "tta.reindex"(%{{.*}}) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 7, 9>}>
  // CHECK-NOT: "tta.advance"
  // CHECK: tt.return %[[R]]
  tt.func @reindex_of_advance_compose(%arg0: !tt.ptr<f32>) -> !tta.addr<f32, 2, 1> {
    %base = tta.make_addr %arg0 to sizes: [16, 16], strides: [16, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
    %a0 = "tta.advance"(%base) <{static_deltas = array<i64: 2, 3>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    %r1 = "tta.reindex"(%a0) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 5, 6>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    tt.return %r1 : !tta.addr<f32, 2, 1>
  }
}

// -----

module {
  // CHECK-LABEL: tt.func @reindex_chain_compose_dynamic(
  // CHECK: %[[ADD0:.*]] = arith.addi %arg1, %arg3 : index
  // CHECK: %[[ADD1:.*]] = arith.addi %arg2, %arg4 : index
  // CHECK: %[[R:.*]] = "tta.reindex"(%[[BASE:.*]], %[[ADD0]], %[[ADD1]])
  // CHECK: tt.return %[[R]]
  tt.func @reindex_chain_compose_dynamic(%arg0: !tt.ptr<f32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> !tta.addr<f32, 2, 1> {
    %base = tta.make_addr %arg0 to sizes: [8, 8], strides: [8, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
    %r0 = "tta.reindex"(%base, %arg1, %arg2) <{operandSegmentSizes = array<i32: 1, 0, 2, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808>}> : (!tta.addr<f32, 2, 1>, index, index) -> !tta.addr<f32, 2, 1>
    %r1 = "tta.reindex"(%r0, %arg3, %arg4) <{operandSegmentSizes = array<i32: 1, 0, 2, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808>}> : (!tta.addr<f32, 2, 1>, index, index) -> !tta.addr<f32, 2, 1>
    tt.return %r1 : !tta.addr<f32, 2, 1>
  }
}

// -----

module {
  // CHECK-LABEL: tt.func @advance_zero_fold(
  // CHECK: %[[BASE:.*]] = tta.make_addr %arg0
  // CHECK-NOT: "tta.advance"
  // CHECK: tt.return %{{.*}}
  tt.func @advance_zero_fold(%arg0: !tt.ptr<f32>) -> !tta.addr<f32, 2, 1> {
    %base = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
    %a = "tta.advance"(%base) <{static_deltas = array<i64: 0, 0>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    tt.return %a : !tta.addr<f32, 2, 1>
  }
}

// -----

module {
  // CHECK-LABEL: tt.func @reindex_chain_compose_static(
  // CHECK: %[[BASE:.*]] = tta.make_addr %arg0
  // CHECK: %[[R:.*]] = "tta.reindex"(%{{.*}}) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 3, 5>}>
  // CHECK-NOT: "tta.reindex"(%[[R]])
  // CHECK: tt.return %[[R]]
  tt.func @reindex_chain_compose_static(%arg0: !tt.ptr<f32>) -> !tta.addr<f32, 2, 1> {
    %base = tta.make_addr %arg0 to sizes: [8, 8], strides: [8, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
    %r0 = "tta.reindex"(%base) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 1, 2>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    %r1 = "tta.reindex"(%r0) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 2, 3>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    tt.return %r1 : !tta.addr<f32, 2, 1>
  }
}

// -----

module {
  // CHECK-LABEL: tt.func @advance_chain_compose_static(
  // CHECK: %[[BASE:.*]] = tta.make_addr %arg0
  // CHECK: %[[A:.*]] = "tta.advance"(%{{.*}}) <{static_deltas = array<i64: 7, 10>}>
  // CHECK-NOT: "tta.advance"(%[[A]])
  // CHECK: tt.return %[[A]]
  tt.func @advance_chain_compose_static(%arg0: !tt.ptr<f32>) -> !tta.addr<f32, 2, 1> {
    %base = tta.make_addr %arg0 to sizes: [8, 8], strides: [8, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
    %a0 = "tta.advance"(%base) <{static_deltas = array<i64: 4, 6>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    %a1 = "tta.advance"(%a0) <{static_deltas = array<i64: 3, 4>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    tt.return %a1 : !tta.addr<f32, 2, 1>
  }
}

// -----

module {
  // CHECK-LABEL: tt.func @reindex_chain_not_compose_indirect(
  // CHECK: %[[R0:.*]] = "tta.reindex"(%[[BASE:.*]], %[[IDX:.*]], %[[MASK:.*]])
  // CHECK: %[[R1:.*]] = "tta.reindex"(%[[R0]])
  // CHECK: tt.return %[[R1]]
  tt.func @reindex_chain_not_compose_indirect(%arg0: !tt.ptr<f32>) -> !tta.addr<f32, 2, 1> {
    %idx = arith.constant dense<[0, 1]> : tensor<2xi32>
    %mask = arith.constant dense<[true, false]> : tensor<2xi1>
    %base = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
    %r0 = "tta.reindex"(%base, %idx, %mask) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0, 0>}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>, tensor<2xi1>) -> !tta.addr<f32, 2, 1>
    %r1 = "tta.reindex"(%r0) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 1, 0>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    tt.return %r1 : !tta.addr<f32, 2, 1>
  }
}
