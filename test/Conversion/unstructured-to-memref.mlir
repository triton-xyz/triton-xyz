// RUN: triton-shared-opt --split-input-file --unstructured-to-memref %s | FileCheck %s

module {
  tt.func public @scalar_gather_scatter(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %off: i32) {
    %val = tts.gather %src[%off] : (<f32>, i32) -> f32
    tts.scatter %val into %dst[%off] : f32 into (<f32>, i32)
    tt.return
  }
}

// CHECK-LABEL: tt.func public @scalar_gather_scatter
// CHECK-SAME: (%[[SRC:.*]]: !tt.ptr<f32>, %[[DST:.*]]: !tt.ptr<f32>, %[[OFF:.*]]: i32)
// CHECK-DAG: %[[SRC_CAST:.*]] = builtin.unrealized_conversion_cast %[[SRC]] : !tt.ptr<f32> to memref<*xf32>
// CHECK-DAG: %[[DST_CAST:.*]] = builtin.unrealized_conversion_cast %[[DST]] : !tt.ptr<f32> to memref<*xf32>
// CHECK: %[[IDX0:.*]] = arith.index_cast %[[OFF]] : i32 to index
// CHECK: %[[SRC_VIEW:.*]] = memref.reinterpret_cast %[[SRC_CAST]] to offset: [%[[IDX0]]]{{.*}} : memref<*xf32> to memref<1xf32{{.*}}>
// CHECK: %[[LOAD:.*]] = affine.load %[[SRC_VIEW]]
// CHECK: %[[IDX1:.*]] = arith.index_cast %[[OFF]] : i32 to index
// CHECK: %[[DST_VIEW:.*]] = memref.reinterpret_cast %[[DST_CAST]] to offset: [%[[IDX1]]]{{.*}} : memref<*xf32> to memref<1xf32{{.*}}>
// CHECK: affine.store %[[LOAD]], %[[DST_VIEW]]

// -----

module {
  tt.func public @gather_no_mask(%src: !tt.ptr<f32>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %val = tts.gather %src[%offsets] : (<f32>, tensor<4xi32>) -> tensor<4xf32>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @gather_no_mask
// CHECK-SAME: (%[[SRC:.*]]: !tt.ptr<f32>)
// CHECK-DAG: %[[OFFSETS:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
// CHECK-DAG: %[[SRC_CAST:.*]] = builtin.unrealized_conversion_cast %[[SRC]] : !tt.ptr<f32> to memref<*xf32>
// CHECK: %[[SRC_MEMREF:.*]] = memref.cast %[[SRC_CAST]] : memref<*xf32> to memref<?xf32>
// CHECK: %[[BASE:.*]] = bufferization.to_tensor %[[SRC_MEMREF]] restrict : memref<?xf32> to tensor<?xf32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4xf32>
// CHECK: linalg.generic {{.*}}ins(%[[OFFSETS]] : tensor<4xi32>) outs(%[[EMPTY]] : tensor<4xf32>)
// CHECK: tensor.extract %[[BASE]]
// CHECK: linalg.yield

// -----

module {
  tt.func public @gather_mask_no_other(%src: !tt.ptr<f32>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %mask = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
    %val = tts.gather %src[%offsets] mask = %mask : (<f32>, tensor<4xi32>) -> tensor<4xf32>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @gather_mask_no_other
// CHECK-SAME: (%[[SRC:.*]]: !tt.ptr<f32>)
// CHECK-DAG: %[[OFFSETS:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
// CHECK-DAG: %[[MASK:.*]] = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
// CHECK: linalg.generic {{.*}}ins(%[[OFFSETS]], %[[MASK]] : tensor<4xi32>, tensor<4xi1>)
// CHECK: scf.if
// CHECK: } else {
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: scf.yield %[[ZERO]] : f32

// -----

module {
  tt.func public @gather_mask_with_other(%src: !tt.ptr<f32>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %mask = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
    %other = arith.constant 1.250000e+00 : f32
    %val = tts.gather %src[%offsets] mask = %mask default = %other : (<f32>, tensor<4xi32>) -> tensor<4xf32>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @gather_mask_with_other
// CHECK-SAME: (%[[SRC:.*]]: !tt.ptr<f32>)
// CHECK-DAG: %[[OFFSETS:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
// CHECK-DAG: %[[MASK:.*]] = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
// CHECK-DAG: %[[OTHER:.*]] = arith.constant 1.250000e+00 : f32
// CHECK: linalg.generic {{.*}}ins(%[[OFFSETS]], %[[MASK]] : tensor<4xi32>, tensor<4xi1>)
// CHECK: scf.if
// CHECK: } else {
// CHECK: scf.yield %[[OTHER]] : f32

// -----

module {
  tt.func public @scatter_no_mask(%dst: !tt.ptr<f32>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %values = arith.constant dense<1.000000e+00> : tensor<4xf32>
    tts.scatter %values into %dst[%offsets] : tensor<4xf32> into (<f32>, tensor<4xi32>)
    tt.return
  }
}

// CHECK-LABEL: tt.func public @scatter_no_mask
// CHECK-SAME: (%[[DST:.*]]: !tt.ptr<f32>)
// CHECK-DAG: %[[OFFSETS:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
// CHECK-DAG: %[[VALUES:.*]] = arith.constant dense<1.000000e+00> : tensor<4xf32>
// CHECK-DAG: %[[DST_CAST:.*]] = builtin.unrealized_conversion_cast %[[DST]] : !tt.ptr<f32> to memref<*xf32>
// CHECK: %[[DST_MEMREF:.*]] = memref.cast %[[DST_CAST]] : memref<*xf32> to memref<?xf32>
// CHECK: linalg.generic {{.*}}ins(%[[OFFSETS]], %[[VALUES]] : tensor<4xi32>, tensor<4xf32>)
// CHECK: memref.store

// -----

module {
  tt.func public @scatter_mask(%dst: !tt.ptr<f32>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %values = arith.constant dense<1.000000e+00> : tensor<4xf32>
    %mask = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
    tts.scatter %values into %dst[%offsets] mask = %mask : tensor<4xf32> into (<f32>, tensor<4xi32>)
    tt.return
  }
}

// CHECK-LABEL: tt.func public @scatter_mask
// CHECK-SAME: (%[[DST:.*]]: !tt.ptr<f32>)
// CHECK-DAG: %[[OFFSETS:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
// CHECK-DAG: %[[VALUES:.*]] = arith.constant dense<1.000000e+00> : tensor<4xf32>
// CHECK-DAG: %[[MASK:.*]] = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
// CHECK-DAG: %[[DST_CAST:.*]] = builtin.unrealized_conversion_cast %[[DST]] : !tt.ptr<f32> to memref<*xf32>
// CHECK: %[[DST_MEMREF:.*]] = memref.cast %[[DST_CAST]] : memref<*xf32> to memref<?xf32>
// CHECK: linalg.generic {{.*}}ins(%[[OFFSETS]], %[[VALUES]], %[[MASK]] : tensor<4xi32>, tensor<4xf32>, tensor<4xi1>)
// CHECK: scf.if
// CHECK: memref.store
