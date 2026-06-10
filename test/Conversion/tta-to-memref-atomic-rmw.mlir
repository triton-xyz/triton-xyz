// RUN: triton-xyz-opt --split-input-file --tta-normalize --tta-to-memref %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @atomic_fadd(
// CHECK-SAME:      %[[ARG0:[-0-9A-Za-z$._]+]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[-0-9A-Za-z$._]+]]: i32,
// CHECK-SAME:      %[[ARG2:[-0-9A-Za-z$._]+]]: f32) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xf32> to memref<?xf32>
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG1]] : i32 to index
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[INDEX_CAST_0]], %[[CONSTANT_0]] : index
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[CONSTANT_1]], %[[MULI_0]] : index
// CHECK:           %[[ATOMIC_RMW_0:.*]] = memref.atomic_rmw addf %[[ARG2]], %[[CAST_0]]{{\[}}%[[ADDI_0]]] : (f32, memref<?xf32>) -> f32
// CHECK:           tt.return
// CHECK:         }
  tt.func @atomic_fadd(%ptr: !tt.ptr<f32>, %off: i32, %val: f32) {
    %addr = tta.make_addr %ptr to sizes: [1], strides: [1], offsets: [0], wrap_boundaries: [0], layout: "strided" : <f32> to !tta.addr<f32, 1, 1>
    %old = "tta.atomic"(%addr, %off, %val) <{kind = "fadd"}> : (!tta.addr<f32, 1, 1>, i32, f32) -> f32
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @masked_atomic_add(
// CHECK-SAME:      %[[ARG0:[-0-9A-Za-z$._]+]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG1:[-0-9A-Za-z$._]+]]: i32,
// CHECK-SAME:      %[[ARG2:[-0-9A-Za-z$._]+]]: i32,
// CHECK-SAME:      %[[ARG3:[-0-9A-Za-z$._]+]]: i1) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<i32> to memref<*xi32>
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xi32> to memref<?xi32>
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG1]] : i32 to index
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[INDEX_CAST_0]], %[[CONSTANT_0]] : index
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[CONSTANT_1]], %[[MULI_0]] : index
// CHECK:           %[[IF_0:.*]] = scf.if %[[ARG3]] -> (i32) {
// CHECK:             %[[ATOMIC_RMW_0:.*]] = memref.atomic_rmw addi %[[ARG2]], %[[CAST_0]]{{\[}}%[[ADDI_0]]] : (i32, memref<?xi32>) -> i32
// CHECK:             scf.yield %[[ATOMIC_RMW_0]] : i32
// CHECK:           } else {
// CHECK:             %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// CHECK:             scf.yield %[[CONSTANT_2]] : i32
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @masked_atomic_add(%ptr: !tt.ptr<i32>, %off: i32, %val: i32, %mask: i1) {
    %addr = tta.make_addr %ptr to sizes: [1], strides: [1], offsets: [0], wrap_boundaries: [0], layout: "strided" : <i32> to !tta.addr<i32, 1, 1>
    %old = "tta.atomic"(%addr, %off, %val, %mask) <{kind = "add"}> : (!tta.addr<i32, 1, 1>, i32, i32, i1) -> i32
    %use = arith.addi %old, %val : i32
    tt.return
  }
}
