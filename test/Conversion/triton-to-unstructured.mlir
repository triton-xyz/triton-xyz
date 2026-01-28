// RUN: triton-xyz-opt --split-input-file --triton-to-unstructured %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func public @masked_gather_scatter(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[CONSTANT_1]] : i32 -> tensor<4xi32>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[SPLAT_0]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           %[[SPLAT_1:.*]] = tt.splat %[[CONSTANT_0]] : i32 -> tensor<4xi32>
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[SPLAT_1]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           %[[SPLAT_2:.*]] = tt.splat %[[ARG2]] : i32 -> tensor<4xi32>
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi slt, %[[MAKE_RANGE_0]], %[[SPLAT_2]] : tensor<4xi32>
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[SPLAT_3:.*]] = tt.splat %[[CONSTANT_2]] : f32 -> tensor<4xf32>
// CHECK:           %[[GATHER_0:.*]] = tts.gather %[[ARG0]]{{\[}}%[[ADDI_0]]] mask = %[[CMPI_0]] default = %[[CONSTANT_2]] : (<f32>, tensor<4xi32>) -> tensor<4xf32>
// CHECK:           tts.scatter %[[GATHER_0]] into %[[ARG1]]{{\[}}%[[ADDI_1]]] mask = %[[CMPI_0]] : tensor<4xf32> into (<f32>, tensor<4xi32>)
// CHECK:           tt.return
// CHECK:         }
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
// CHECK-LABEL:   tt.func public @offset_width_upgrade(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 5 : i64
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[CONSTANT_2]] : i64 -> tensor<4xi64>
// CHECK:           %[[SPLAT_1:.*]] = tt.splat %[[CONSTANT_1]] : i32 -> tensor<4xi32>
// CHECK:           %[[EXTSI_0:.*]] = arith.extsi %[[SPLAT_1]] : tensor<4xi32> to tensor<4xi64>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[EXTSI_0]], %[[SPLAT_0]] : tensor<4xi64>
// CHECK:           %[[EXTSI_1:.*]] = arith.extsi %[[MAKE_RANGE_0]] : tensor<4xi32> to tensor<4xi64>
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[ADDI_0]], %[[EXTSI_1]] : tensor<4xi64>
// CHECK:           %[[GATHER_0:.*]] = tts.gather %[[ARG0]]{{\[}}%[[ADDI_1]]] : (<f32>, tensor<4xi64>) -> tensor<4xf32>
// CHECK:           %[[SPLAT_2:.*]] = tt.splat %[[CONSTANT_0]] : i32 -> tensor<4xi32>
// CHECK:           %[[ADDI_2:.*]] = arith.addi %[[SPLAT_2]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           tts.scatter %[[GATHER_0]] into %[[ARG1]]{{\[}}%[[ADDI_2]]] : tensor<4xf32> into (<f32>, tensor<4xi32>)
// CHECK:           tt.return
// CHECK:         }
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
// CHECK-LABEL:   tt.func public @loop_ptr_iter_args(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[CONSTANT_1]] : i32 -> tensor<4xi32>
// CHECK:           %[[SPLAT_1:.*]] = tt.splat %[[CONSTANT_0]] : i32 -> tensor<4xi32>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[SPLAT_0]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[SPLAT_1]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[ARG2]] step %[[CONSTANT_3]] iter_args(%[[VAL_1:.*]] = %[[ADDI_0]], %[[VAL_2:.*]] = %[[ADDI_1]]) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
// CHECK:             %[[GATHER_0:.*]] = tts.gather %[[ARG0]]{{\[}}%[[VAL_1]]] : (<f32>, tensor<4xi32>) -> tensor<4xf32>
// CHECK:             tts.scatter %[[GATHER_0]] into %[[ARG1]]{{\[}}%[[VAL_2]]] : tensor<4xf32> into (<f32>, tensor<4xi32>)
// CHECK:             %[[ADDI_2:.*]] = arith.addi %[[VAL_1]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:             %[[ADDI_3:.*]] = arith.addi %[[VAL_2]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:             scf.yield %[[ADDI_2]], %[[ADDI_3]] : tensor<4xi32>, tensor<4xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
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
// CHECK-LABEL:   tt.func public @make_tensor_ptr_add_base(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f16>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f16>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[GET_PROGRAM_ID_0:.*]] = tt.get_program_id x : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 8 : i32
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[GET_PROGRAM_ID_0]], %[[CONSTANT_2]] : i32
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[CONSTANT_1]], %[[MULI_0]] : i32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 4 : i64
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : i64
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_6:.*]] = arith.constant 0 : i32
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[ADDI_0]], %[[CONSTANT_5]] : i32
// CHECK:           %[[MAKE_TENSOR_PTR_0:.*]] = tt.make_tensor_ptr %[[ARG0]], {{\[}}%[[CONSTANT_3]]], {{\[}}%[[CONSTANT_4]]], {{\[}}%[[ADDI_1]]] {order = array<i32: 0>} : <tensor<4xf16>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[MAKE_TENSOR_PTR_0]] : !tt.ptr<tensor<4xf16>>
// CHECK:           %[[ADDI_2:.*]] = arith.addi %[[CONSTANT_0]], %[[CONSTANT_6]] : i32
// CHECK:           %[[MAKE_TENSOR_PTR_1:.*]] = tt.make_tensor_ptr %[[ARG1]], {{\[}}%[[CONSTANT_3]]], {{\[}}%[[CONSTANT_4]]], {{\[}}%[[ADDI_2]]] {order = array<i32: 0>} : <tensor<4xf16>>
// CHECK:           tt.store %[[MAKE_TENSOR_PTR_1]], %[[LOAD_0]] : !tt.ptr<tensor<4xf16>>
// CHECK:           tt.return
// CHECK:         }
  tt.func public @make_tensor_ptr_add_base(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>) {
    %pid = tt.get_program_id x : i32
    %c8_i32 = arith.constant 8 : i32
    %base_off = arith.muli %pid, %c8_i32 : i32
    %base_ptr = tt.addptr %arg0, %base_off : !tt.ptr<f16>, i32
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %tptr = tt.make_tensor_ptr %base_ptr, [%c4_i64], [%c1_i64], [%c1_i32] {order = array<i32: 0>} : <tensor<4xf16>>
    %val = tt.load %tptr : !tt.ptr<tensor<4xf16>>
    %out_tptr = tt.make_tensor_ptr %arg1, [%c4_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<4xf16>>
    tt.store %out_tptr, %val : !tt.ptr<tensor<4xf16>>
    tt.return
  }
}
