// RUN: triton-xyz-opt --split-input-file --tta-normalize --tta-to-memref %s | FileCheck %s

module {
// CHECK-LABEL: tt.func @atomic_fadd(
// CHECK-SAME:    %[[PTR:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:    %[[OFF:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:    %[[VAL:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32) {
// CHECK:        %[[CAST_PTR:.*]] = builtin.unrealized_conversion_cast %[[PTR]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:        %[[MEMREF:.*]] = memref.cast %[[CAST_PTR]] : memref<*xf32> to memref<?xf32>
// CHECK:        %[[IDX:.*]] = arith.index_cast %[[OFF]] : i32 to index
// CHECK:        %[[OFFSET:.*]] = arith.addi {{.*}} : index
// CHECK:        memref.atomic_rmw addf %[[VAL]], %[[MEMREF]]{{\[}}%[[OFFSET]]] : (f32, memref<?xf32>) -> f32
// CHECK-NOT:    memref.generic_atomic_rmw
// CHECK:        tt.return
  tt.func @atomic_fadd(%ptr: !tt.ptr<f32>, %off: i32, %val: f32) {
    %addr = tta.make_addr %ptr to sizes: [1], strides: [1], offsets: [0], wrap_boundaries: [0], layout: "strided" : <f32> to !tta.addr<f32, 1, 1>
    %old = "tta.atomic"(%addr, %off, %val) <{kind = "fadd"}> : (!tta.addr<f32, 1, 1>, i32, f32) -> f32
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @masked_atomic_add(
// CHECK-SAME:    %[[PTR:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:    %[[OFF:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:    %[[VAL:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:    %[[MASK:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i1) {
// CHECK:        %[[CAST_PTR:.*]] = builtin.unrealized_conversion_cast %[[PTR]] : !tt.ptr<i32> to memref<*xi32>
// CHECK:        %[[MEMREF:.*]] = memref.cast %[[CAST_PTR]] : memref<*xi32> to memref<?xi32>
// CHECK:        %[[IDX:.*]] = arith.index_cast %[[OFF]] : i32 to index
// CHECK:        %[[OFFSET:.*]] = arith.addi {{.*}} : index
// CHECK:        %[[OLD:.*]] = scf.if %[[MASK]] -> (i32) {
// CHECK:          %[[ATOMIC:.*]] = memref.atomic_rmw addi %[[VAL]], %[[MEMREF]]{{\[}}%[[OFFSET]]] : (i32, memref<?xi32>) -> i32
// CHECK:          scf.yield %[[ATOMIC]] : i32
// CHECK:        } else {
// CHECK:          %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:          scf.yield %[[ZERO]] : i32
// CHECK:        }
// CHECK-NOT:    memref.generic_atomic_rmw
// CHECK:        tt.return
  tt.func @masked_atomic_add(%ptr: !tt.ptr<i32>, %off: i32, %val: i32, %mask: i1) {
    %addr = tta.make_addr %ptr to sizes: [1], strides: [1], offsets: [0], wrap_boundaries: [0], layout: "strided" : <i32> to !tta.addr<i32, 1, 1>
    %old = "tta.atomic"(%addr, %off, %val, %mask) <{kind = "add"}> : (!tta.addr<i32, 1, 1>, i32, i32, i1) -> i32
    %use = arith.addi %old, %val : i32
    tt.return
  }
}
