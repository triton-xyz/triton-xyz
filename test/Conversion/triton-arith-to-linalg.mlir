// RUN: triton-shared-opt --triton-arith-to-linalg --split-input-file %s | FileCheck %s

module {
  tt.func @program_info() -> i32 {
    %pid = tt.get_program_id x : i32
    %nprog = tt.get_num_programs x : i32
    %sum = arith.addi %pid, %nprog : i32
    tt.return %sum : i32
  }
}

// CHECK-LABEL: func.func @program_info(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) -> i32
// CHECK: arith.addi %arg
// CHECK-NOT: tt.get_program_id
// CHECK-NOT: tt.get_num_programs

// -----

module {
  tt.func @broadcast_transpose(%arg0: f32) -> tensor<4x2xf32> {
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %exp = tt.expand_dims %range {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %bcast = tt.broadcast %exp : tensor<1x4xi32> -> tensor<2x4xi32>
    %vals = arith.sitofp %bcast : tensor<2x4xi32> to tensor<2x4xf32>
    %splat = tt.splat %arg0 : f32 -> tensor<2x4xf32>
    %sum = arith.addf %vals, %splat : tensor<2x4xf32>
    %t = tt.trans %sum {order = array<i32: 1, 0>} : tensor<2x4xf32> -> tensor<4x2xf32>
    tt.return %t : tensor<4x2xf32>
  }
}

// CHECK-LABEL: func.func @broadcast_transpose
// CHECK: linalg.index 0
// CHECK: tensor.expand_shape
// CHECK: broadcastDims
// CHECK: linalg.transpose {{.*}} permutation = [1, 0]

// -----

module {
  tt.func @reshape_collapse(%arg0: tensor<2x2xf32>) -> tensor<4xf32> {
    %reshaped = tt.reshape %arg0 allow_reorder : tensor<2x2xf32> -> tensor<4xf32>
    tt.return %reshaped : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @reshape_collapse
// CHECK: tensor.collapse_shape

// -----

module {
  tt.func @bitcast_tensor(%arg0: tensor<4xi32>) -> tensor<4xf32> {
    %cast = tt.bitcast %arg0 : tensor<4xi32> -> tensor<4xf32>
    tt.return %cast : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @bitcast_tensor
// CHECK: linalg.generic
// CHECK: arith.bitcast

// -----

module {
  tt.func @extern_unary(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %res = tt.extern_elementwise %arg0 {libname = "", libpath = "", pure = true, symbol = "__nv_sqrtf"} : (tensor<4xf32>) -> tensor<4xf32>
    tt.return %res : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @extern_unary
// CHECK: linalg.generic
// CHECK: math.sqrt

// -----

module {
  tt.func @cumsum_1d() -> tensor<4xf32> {
    %c1 = arith.constant 1.0 : f32
    %input = tt.splat %c1 : f32 -> tensor<4xf32>
    %res = "tt.scan"(%input) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg0: f32, %arg1: f32):
      %sum = arith.addf %arg0, %arg1 : f32
      tt.scan.return %sum : f32
    }) : (tensor<4xf32>) -> tensor<4xf32>
    tt.return %res : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @cumsum_1d
// CHECK: ttx.cumsum

// -----

module {
  tt.func @reduce_add() -> f32 {
    %c1 = arith.constant 1.0 : f32
    %input = tt.splat %c1 : f32 -> tensor<4xf32>
    %res = "tt.reduce"(%input) ({
    ^bb0(%arg0: f32, %arg1: f32):
      %sum = arith.addf %arg0, %arg1 : f32
      tt.reduce.return %sum : f32
    }) {axis = 0 : i32} : (tensor<4xf32>) -> f32
    tt.return %res : f32
  }
}

// CHECK-LABEL: func.func @reduce_add
// CHECK: linalg.reduce
// CHECK: arith.addf

// -----

module {
  tt.func @dot_2x2_2x2() -> tensor<2x2xf32> {
    %c1 = arith.constant 1.0 : f32
    %a = tt.splat %c1 : f32 -> tensor<2x2xf32>
    %b = tt.splat %c1 : f32 -> tensor<2x2xf32>
    %c0 = arith.constant 0.0 : f32
    %c = tt.splat %c0 : f32 -> tensor<2x2xf32>
    %res = tt.dot %a, %b, %c {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<2x2xf32> * tensor<2x2xf32> -> tensor<2x2xf32>
    tt.return %res : tensor<2x2xf32>
  }
}

// CHECK-LABEL: func.func @dot_2x2_2x2
// CHECK: linalg.matmul

// -----

module {
  tt.func @split_join(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %lhs, %rhs = tt.split %arg0 : tensor<2x2xf32> -> tensor<2xf32>
    %joined = tt.join %lhs, %rhs : tensor<2xf32> -> tensor<2x2xf32>
    tt.return %joined : tensor<2x2xf32>
  }
}

// CHECK-LABEL: func.func @split_join
// CHECK: tensor.extract_slice
// CHECK: tensor.insert_slice

// -----

module {
  tt.func @cat_dim0() -> tensor<2x2xf32> {
    %c1 = arith.constant 1.0 : f32
    %c2 = arith.constant 2.0 : f32
    %a = tt.splat %c1 : f32 -> tensor<1x2xf32>
    %b = tt.splat %c2 : f32 -> tensor<1x2xf32>
    %cat = tt.cat %a, %b : tensor<1x2xf32> -> tensor<2x2xf32>
    tt.return %cat : tensor<2x2xf32>
  }
}

// CHECK-LABEL: func.func @cat_dim0
// CHECK: tensor.insert_slice

// -----

module {
  tt.func @assert_scalar(%arg0: i32) {
    %zero = arith.constant 0 : i32
    %cond = arith.cmpi sgt, %arg0, %zero : i32
    tt.assert %cond, "x > 0" : i1
    tt.return
  }
}

// CHECK-LABEL: func.func @assert_scalar
// CHECK: cf.assert
// CHECK-NOT: tt.assert

// -----

module {
  tt.func @dense_constant() -> tensor<2xf32> {
    %cst = arith.constant dense<1.0> : tensor<2xf32>
    tt.return %cst : tensor<2xf32>
  }
}

// CHECK-LABEL: func.func @dense_constant
// CHECK: linalg.fill

// -----

module {
  tt.func @call_callee(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    tt.return %arg0 : tensor<4xf32>
  }

  tt.func @call_caller(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %res = tt.call @call_callee(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
    tt.return %res : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @call_caller
// CHECK: call @call_callee

// -----

module {
  tt.func @fptofp_rtne(%arg0: tensor<4xf32>) -> tensor<4xf16> {
    %res = tt.fp_to_fp %arg0, rounding = rtne : tensor<4xf32> -> tensor<4xf16>
    tt.return %res : tensor<4xf16>
  }
}

// CHECK-LABEL: func.func @fptofp_rtne
// CHECK: arith.truncf

// -----

module {
  tt.func @clamp_all(%x: tensor<4xf32>, %min: tensor<4xf32>,
                     %max: tensor<4xf32>) -> tensor<4xf32> {
    %res = tt.clampf %x, %min, %max, propagateNan = all : tensor<4xf32>
    tt.return %res : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @clamp_all
// CHECK: arith.maximumf
// CHECK: arith.minimumf

// -----

module {
  tt.func @precise_math(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %sqrt = tt.precise_sqrt %arg0 : tensor<4xf32>
    %div = tt.precise_divf %sqrt, %arg1 : tensor<4xf32>
    tt.return %div : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @precise_math
// CHECK: math.sqrt
// CHECK: arith.divf

// -----

module {
  tt.func @mulhiui(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
    %res = tt.mulhiui %arg0, %arg1 : tensor<4xi32>
    tt.return %res : tensor<4xi32>
  }
}

// CHECK-LABEL: func.func @mulhiui
// CHECK: arith.mului_extended

// -----

module {
  tt.func @unsplat_scalar(%arg0: tensor<1xf32>) -> f32 {
    %res = tt.unsplat %arg0 : tensor<1xf32>
    tt.return %res : f32
  }
}

// CHECK-LABEL: func.func @unsplat_scalar
// CHECK: tensor.extract

// -----

module {
  tt.func @argmax_4() -> i32 {
    %vals = arith.constant dense<0.0> : tensor<4xf32>
    %idx = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %res:2 = "tt.reduce"(%vals, %idx) <{axis = 0 : i32}> ({
    ^bb0(%v: f32, %i: i32, %v_acc: f32, %i_acc: i32):
      %eq = arith.cmpf oeq, %v, %v_acc : f32
      %lt = arith.cmpi slt, %i, %i_acc : i32
      %tie = arith.andi %eq, %lt : i1
      %gt = arith.cmpf ogt, %v, %v_acc : f32
      %pick = arith.ori %gt, %tie : i1
      %v_out = arith.select %pick, %v, %v_acc : f32
      %i_out = arith.select %pick, %i, %i_acc : i32
      tt.reduce.return %v_out, %i_out : f32, i32
    }) : (tensor<4xf32>, tensor<4xi32>) -> (f32, i32)
    tt.return %res#1 : i32
  }
}

// CHECK-LABEL: func.func @argmax_4
// CHECK: linalg.reduce
// CHECK: arith.cmpf ogt

// -----

module {
  tt.func @argmin_4() -> i32 {
    %vals = arith.constant dense<0.0> : tensor<4xf32>
    %idx = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %res:2 = "tt.reduce"(%vals, %idx) <{axis = 0 : i32}> ({
    ^bb0(%v: f32, %i: i32, %v_acc: f32, %i_acc: i32):
      %eq = arith.cmpf oeq, %v, %v_acc : f32
      %lt = arith.cmpi slt, %i, %i_acc : i32
      %tie = arith.andi %eq, %lt : i1
      %ltv = arith.cmpf olt, %v, %v_acc : f32
      %pick = arith.ori %ltv, %tie : i1
      %v_out = arith.select %pick, %v, %v_acc : f32
      %i_out = arith.select %pick, %i, %i_acc : i32
      tt.reduce.return %v_out, %i_out : f32, i32
    }) : (tensor<4xf32>, tensor<4xi32>) -> (f32, i32)
    tt.return %res#1 : i32
  }
}

// CHECK-LABEL: func.func @argmin_4
// CHECK: linalg.reduce
// CHECK: arith.cmpf olt

// -----

module {
  tt.func @minmax_select(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %cmp = arith.cmpf ogt, %arg0, %arg1 : tensor<4xf32>
    %sel = arith.select %cmp, %arg0, %arg1 : tensor<4xi1>, tensor<4xf32>
    tt.return %sel : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @minmax_select
// CHECK: arith.maximumf

// -----

module {
  tt.func @extern_binary(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %res = tt.extern_elementwise %arg0, %arg1 {libname = "", libpath = "", pure = true, symbol = "__nv_atan2f"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    tt.return %res : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @extern_binary
// CHECK: math.atan2
