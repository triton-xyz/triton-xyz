// RUN: triton-xyz-opt --split-input-file --tta-normalize --tta-to-memref --canonicalize --cse %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @block_atomic_add_scalar(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[SRC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<i32> to memref<*xi32>
// CHECK:           %[[CAST:.*]] = memref.cast %[[SRC]] : memref<*xi32> to memref<?xi32>
// CHECK:           %[[OFFSET:.*]] = arith.index_cast %[[ARG1]] : i32 to index
// CHECK:           %[[ATOMIC:.*]] = memref.atomic_rmw addi %[[ARG2]], %[[CAST]]{{\[}}%[[OFFSET]]] : (i32, memref<?xi32>) -> i32
// CHECK:           tt.return
// CHECK:         }
  tt.func @block_atomic_add_scalar(%ptr: !tt.ptr<i32>, %off: i32, %val: i32) {
    %ptr_i = tta.make_addr %ptr to sizes: [4], strides: [1], offsets: [0], wrap_boundaries: [0], layout: "block", parent_shape: [4] {layout_payload = {order = array<i32: 0>}} : <i32> to !tta.addr<i32, 1, 1>
    %r = "tta.atomic"(%ptr_i, %off, %val) <{kind = "add"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
    %u = arith.addi %r, %val : i32
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @block_atomic_add_tensor(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>) {
// CHECK:           %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:           %[[OFFSETS:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
// CHECK:           %[[VALUES:.*]] = arith.constant dense<[10, 11, 12, 13]> : tensor<4xi32>
// CHECK:           %[[MASK:.*]] = arith.constant dense<[true, false, true, true]> : tensor<4xi1>
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[EMPTY]]) -> (tensor<4xi32>) {
// CHECK:             %[[OFF:.*]] = tensor.extract %[[OFFSETS]]{{\[}}%[[IV]]] : tensor<4xi32>
// CHECK:             %[[VAL:.*]] = tensor.extract %[[VALUES]]{{\[}}%[[IV]]] : tensor<4xi32>
// CHECK:             %[[LANE_MASK:.*]] = tensor.extract %[[MASK]]{{\[}}%[[IV]]] : tensor<4xi1>
// CHECK:             %[[CASTED:.*]] = arith.index_cast %[[OFF]] : i32 to index
// CHECK:             %[[IF:.*]] = scf.if %[[LANE_MASK]] -> (i32) {
// CHECK:               %[[ATOMIC:.*]] = memref.atomic_rmw addi %[[VAL]], %{{.*}}{{\[}}%[[CASTED]]] : (i32, memref<?xi32>) -> i32
// CHECK:               scf.yield %[[ATOMIC]] : i32
// CHECK:             } else {
// CHECK:               scf.yield %[[ZERO]] : i32
// CHECK:             }
// CHECK:             %[[NEXT:.*]] = tensor.insert %[[IF]] into %[[ACC]]{{\[}}%[[IV]]] : tensor<4xi32>
// CHECK:             scf.yield %[[NEXT]] : tensor<4xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @block_atomic_add_tensor(%ptr: !tt.ptr<i32>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %vals = arith.constant dense<[10, 11, 12, 13]> : tensor<4xi32>
    %mask = arith.constant dense<[true, false, true, true]> : tensor<4xi1>
    %ptr_i = tta.make_addr %ptr to sizes: [4], strides: [1], offsets: [0], wrap_boundaries: [0], layout: "block", parent_shape: [4] {layout_payload = {order = array<i32: 0>}} : <i32> to !tta.addr<i32, 1, 1>
    %r = "tta.atomic"(%ptr_i, %offsets, %vals, %mask) <{kind = "add"}> : (!tta.addr<i32, 1, 1>, tensor<4xi32>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
    %u = arith.addi %r, %vals : tensor<4xi32>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @block_atomic_cas_scalar(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[SRC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<i32> to memref<*xi32>
// CHECK:           %[[CAST:.*]] = memref.cast %[[SRC]] : memref<*xi32> to memref<?xi32>
// CHECK:           %[[OFFSET:.*]] = arith.index_cast %[[ARG1]] : i32 to index
// CHECK:           %[[GENERIC:.*]] = memref.generic_atomic_rmw %[[CAST]]{{\[}}%[[OFFSET]]] : memref<?xi32> {
// CHECK:           ^bb0(%[[CUR:.*]]: i32):
// CHECK:             %[[CMP:.*]] = arith.cmpi eq, %[[CUR]], %[[ARG2]] : i32
// CHECK:             %[[SELECT:.*]] = arith.select %[[CMP]], %[[ARG3]], %[[CUR]] : i32
// CHECK:             memref.atomic_yield %[[SELECT]] : i32
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @block_atomic_cas_scalar(%ptr: !tt.ptr<i32>, %off: i32, %cmp: i32, %val: i32) {
    %ptr_i = tta.make_addr %ptr to sizes: [4], strides: [1], offsets: [0], wrap_boundaries: [0], layout: "block", parent_shape: [4] {layout_payload = {order = array<i32: 0>}} : <i32> to !tta.addr<i32, 1, 1>
    %r = "tta.atomic_cas"(%ptr_i, %off, %cmp, %val) : (!tta.addr<i32, 1, 1>, i32, i32, i32) -> i32
    %u = arith.addi %r, %val : i32
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @block_atomic_cas_tensor(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>) {
// CHECK:           %[[OFFSETS:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
// CHECK:           %[[COMPARE:.*]] = arith.constant dense<[7, 6, 5, 4]> : tensor<4xi32>
// CHECK:           %[[VALUES:.*]] = arith.constant dense<[20, 21, 22, 23]> : tensor<4xi32>
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[EMPTY]]) -> (tensor<4xi32>) {
// CHECK:             %[[OFF:.*]] = tensor.extract %[[OFFSETS]]{{\[}}%[[IV]]] : tensor<4xi32>
// CHECK:             %[[CMP:.*]] = tensor.extract %[[COMPARE]]{{\[}}%[[IV]]] : tensor<4xi32>
// CHECK:             %[[VAL:.*]] = tensor.extract %[[VALUES]]{{\[}}%[[IV]]] : tensor<4xi32>
// CHECK:             %[[CASTED:.*]] = arith.index_cast %[[OFF]] : i32 to index
// CHECK:             %[[GENERIC:.*]] = memref.generic_atomic_rmw %{{.*}}{{\[}}%[[CASTED]]] : memref<?xi32> {
// CHECK:             ^bb0(%[[CUR:.*]]: i32):
// CHECK:               %[[EQ:.*]] = arith.cmpi eq, %[[CUR]], %[[CMP]] : i32
// CHECK:               %[[SELECT:.*]] = arith.select %[[EQ]], %[[VAL]], %[[CUR]] : i32
// CHECK:               memref.atomic_yield %[[SELECT]] : i32
// CHECK:             }
// CHECK:             %[[NEXT:.*]] = tensor.insert %[[GENERIC]] into %[[ACC]]{{\[}}%[[IV]]] : tensor<4xi32>
// CHECK:             scf.yield %[[NEXT]] : tensor<4xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @block_atomic_cas_tensor(%ptr: !tt.ptr<i32>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %cmp = arith.constant dense<[7, 6, 5, 4]> : tensor<4xi32>
    %vals = arith.constant dense<[20, 21, 22, 23]> : tensor<4xi32>
    %ptr_i = tta.make_addr %ptr to sizes: [4], strides: [1], offsets: [0], wrap_boundaries: [0], layout: "block", parent_shape: [4] {layout_payload = {order = array<i32: 0>}} : <i32> to !tta.addr<i32, 1, 1>
    %r = "tta.atomic_cas"(%ptr_i, %offsets, %cmp, %vals) : (!tta.addr<i32, 1, 1>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    %u = arith.addi %r, %vals : tensor<4xi32>
    tt.return
  }
}
