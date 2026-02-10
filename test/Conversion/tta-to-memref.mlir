// RUN: triton-xyz-opt --split-input-file --tta-to-memref --canonicalize --cse %s | FileCheck %s

// CHECK-NOT: tta.
// CHECK-NOT: tts.

module {
// CHECK-LABEL: tt.func @structured_basic(
// CHECK: memref.reinterpret_cast
// CHECK: bufferization.materialize_in_destination
  tt.func @structured_basic(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %src_addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %val = "tta.load"(%src_addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
    %dst_addr = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @block_ptr_with_advance(
// CHECK: memref.reinterpret_cast
// CHECK: memref<4x4xf16
// CHECK: bufferization.materialize_in_destination
  tt.func @block_ptr_with_advance(%base: !tt.ptr<f16>) {
    %addr = tta.make_addr %base to sizes: [4, 4], strides: [4, 1], offsets: [0, 0], shape: [4, 4], order: [1, 0] : <f16> to !tta.addr<f16, 2, 1>
    %next = "tta.advance"(%addr) <{static_deltas = array<i64: 0, 1>}> : (!tta.addr<f16, 2, 1>) -> !tta.addr<f16, 2, 1>
    %val = "tta.load"(%next) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f16, 2, 1>) -> tensor<4x4xf16>
    "tta.store"(%addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f16, 2, 1>, tensor<4x4xf16>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @indirect_masked_load_store(
// CHECK: scf.for
// CHECK: tensor.extract
// CHECK: bufferization.materialize_in_destination
  tt.func @indirect_masked_load_store(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %mask = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
    %other = arith.constant 1.250000e+00 : f32

    %src_addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %src_idx = "tta.reindex"(%src_addr, %offsets, %mask) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 1, 1>
    %val = "tta.load"(%src_idx, %other) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, f32) -> tensor<4xf32>

    %dst_addr = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %dst_idx = "tta.reindex"(%dst_addr, %offsets, %mask) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 1, 1>
    "tta.store"(%dst_idx, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
    tt.return
  }
}


// -----

module {
// CHECK-LABEL: tt.func @imported_addr_basic(
// CHECK: memref.reinterpret_cast
// CHECK: bufferization.materialize_in_destination
  tt.func @imported_addr_basic(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %src_addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %val = "tta.load"(%src_addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>

    %dst_addr = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
    tt.return
  }
}


// -----

module {
// CHECK-LABEL: tt.func @loop_carried_addr_supported(
// CHECK: scf.for
// CHECK: memref.reinterpret_cast
// CHECK: memref.copy
// CHECK: memref.reinterpret_cast
// CHECK: bufferization.materialize_in_destination
  tt.func @loop_carried_addr_supported(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %src_addr0 = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %dst_addr0 = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %res:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%src_addr = %src_addr0, %dst_addr = %dst_addr0) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>) : i32 {
      %val = "tta.load"(%src_addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
      "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
      %next_src = "tta.advance"(%src_addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      %next_dst = "tta.advance"(%dst_addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      scf.yield %next_src, %next_dst : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
    }
    tt.return
  }
}


// -----

module {
// CHECK-LABEL: tt.func @loop_carried_addr_supported_non_unit(
// CHECK: scf.for
// CHECK: arith.subi
// CHECK: arith.divsi
// CHECK: memref.reinterpret_cast
// CHECK: bufferization.materialize_in_destination
  tt.func @loop_carried_addr_supported_non_unit(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
    %c3 = arith.constant 3 : i32
    %c2 = arith.constant 2 : i32
    %src_addr0 = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %dst_addr0 = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %res:2 = scf.for %iv = %c3 to %n step %c2 iter_args(%src_addr = %src_addr0, %dst_addr = %dst_addr0) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>) : i32 {
      %val = "tta.load"(%src_addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
      "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
      %next_src = "tta.advance"(%src_addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      %next_dst = "tta.advance"(%dst_addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      scf.yield %next_src, %next_dst : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
    }
    tt.return
  }
}


// -----

module {
// CHECK-LABEL: tt.func @loop_carried_addr_supported_index_iv(
// CHECK: scf.for
// CHECK: arith.subi
// CHECK: arith.divsi
// CHECK: memref.reinterpret_cast
// CHECK: bufferization.materialize_in_destination
  tt.func @loop_carried_addr_supported_index_iv(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: index) {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %src_addr0 = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %dst_addr0 = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %res:2 = scf.for %iv = %c3 to %n step %c2 iter_args(%src_addr = %src_addr0, %dst_addr = %dst_addr0) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>) {
      %val = "tta.load"(%src_addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
      "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
      %next_src = "tta.advance"(%src_addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      %next_dst = "tta.advance"(%dst_addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      scf.yield %next_src, %next_dst : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
    }
    tt.return
  }
}


// -----

module {
// CHECK-LABEL: tt.func @loop_carried_addr_supported_reindex(
// CHECK: scf.for
// CHECK: arith.index_cast
// CHECK: memref.reinterpret_cast
// CHECK: bufferization.materialize_in_destination
  tt.func @loop_carried_addr_supported_reindex(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %src_addr0 = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %dst_addr0 = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %res:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%src_addr = %src_addr0, %dst_addr = %dst_addr0) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>) : i32 {
      %val = "tta.load"(%src_addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
      "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
      %next_src = "tta.reindex"(%src_addr) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      %next_dst = "tta.reindex"(%dst_addr) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      scf.yield %next_src, %next_dst : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
    }
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @indirect_non_zero_offset(
// CHECK: scf.for
// CHECK: arith.addi
// CHECK: bufferization.materialize_in_destination
  tt.func @indirect_non_zero_offset(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>

    %src_addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %src_idx = "tta.reindex"(%src_addr, %offsets) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 1>}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
    %val = "tta.load"(%src_idx) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>

    %dst_addr = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %dst_idx = "tta.reindex"(%dst_addr, %offsets) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 1>}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
    "tta.store"(%dst_idx, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @chained_indirect_same_dim(
// CHECK: arith.index_cast
// CHECK: arith.addi
// CHECK: arith.andi
// CHECK: scf.for
  tt.func @chained_indirect_same_dim(%src: !tt.ptr<f32>, %idx0: tensor<4xi32>, %idx1: tensor<4xi32>, %mask0: tensor<4xi1>, %mask1: tensor<4xi1>) {
    %addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %r0 = "tta.reindex"(%addr, %idx0, %mask0) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 1, 1>
    %r1 = "tta.reindex"(%r0, %idx1, %mask1) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 1, 1>
    %val = "tta.load"(%r1) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>

    %out_addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %out_r0 = "tta.reindex"(%out_addr, %idx0, %mask0) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 1, 1>
    %out_r1 = "tta.reindex"(%out_r0, %idx1, %mask1) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 1, 1>
    "tta.store"(%out_r1, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @chained_indirect_dynamic_static_dim(
// CHECK: tensor.cast
// CHECK: arith.addi
// CHECK: arith.andi
// CHECK: scf.for
  tt.func @chained_indirect_dynamic_static_dim(%src: !tt.ptr<f32>, %idx_dyn: tensor<?xi32>, %idx_static: tensor<4xi32>, %mask_dyn: tensor<?xi1>, %mask_static: tensor<4xi1>) {
    %addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %r0 = "tta.reindex"(%addr, %idx_dyn, %mask_dyn) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>, tensor<?xi32>, tensor<?xi1>) -> !tta.addr<f32, 1, 1>
    %r1 = "tta.reindex"(%r0, %idx_static, %mask_static) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 1, 1>
    %val = "tta.load"(%r1) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>

    %out_addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %out_r0 = "tta.reindex"(%out_addr, %idx_dyn, %mask_dyn) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>, tensor<?xi32>, tensor<?xi1>) -> !tta.addr<f32, 1, 1>
    %out_r1 = "tta.reindex"(%out_r0, %idx_static, %mask_static) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 1, 1>
    "tta.store"(%out_r1, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @atomic_scalar_basic(
// CHECK: memref.generic_atomic_rmw
// CHECK: arith.select
// CHECK: memref.atomic_yield
// CHECK: memref.generic_atomic_rmw
  tt.func @atomic_scalar_basic(%ptr: !tt.ptr<i32>, %off: i32, %val: i32, %mask: i1) {
    %ptr_i = tta.make_addr %ptr to sizes: [1], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to !tta.addr<i32, 1, 1>
    %r0 = "tta.atomic"(%ptr_i, %off, %val, %mask) <{kind = "add"}> : (!tta.addr<i32, 1, 1>, i32, i32, i1) -> i32
    %r1 = "tta.atomic"(%ptr_i, %off, %r0) <{kind = "xchg"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
    %r2 = arith.addi %r0, %r1 : i32
    %r3 = arith.addi %r2, %val : i32
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @atomic_float_add(
// CHECK: memref.generic_atomic_rmw
// CHECK: arith.addf
// CHECK: memref.atomic_yield
  tt.func @atomic_float_add(%ptr: !tt.ptr<f32>, %off: i32, %val: f32) {
    %ptr_i = tta.make_addr %ptr to sizes: [1], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %r = "tta.atomic"(%ptr_i, %off, %val) <{kind = "fadd"}> : (!tta.addr<f32, 1, 1>, i32, f32) -> f32
    %u = arith.addf %r, %val : f32
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @atomic_cas_scalar(
// CHECK: memref.generic_atomic_rmw
// CHECK: arith.cmpi eq
// CHECK: arith.select
// CHECK: memref.atomic_yield
  tt.func @atomic_cas_scalar(%ptr: !tt.ptr<i32>, %off: i32, %cmp: i32, %val: i32) {
    %r = "tta.atomic_cas"(%ptr, %off, %cmp, %val) : (!tt.ptr<i32>, i32, i32, i32) -> i32
    %u = arith.addi %r, %val : i32
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @atomic_cas_float(
// CHECK: memref.generic_atomic_rmw
// CHECK: arith.cmpf oeq
// CHECK: arith.select
// CHECK: memref.atomic_yield
  tt.func @atomic_cas_float(%ptr: !tt.ptr<f32>, %off: i32, %cmp: f32, %val: f32) {
    %r = "tta.atomic_cas"(%ptr, %off, %cmp, %val) : (!tt.ptr<f32>, i32, f32, f32) -> f32
    %u = arith.addf %r, %val : f32
    tt.return
  }
}
