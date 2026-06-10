// RUN: triton-xyz-opt --split-input-file --proton-to-xyz %s | FileCheck %s
// REQUIRES: triton_xyz_build_proton

// -----

module {
// CHECK-LABEL:   func.func private @proton_cpu_record_end(i64)
// CHECK:         func.func private @proton_cpu_record_start(i64)
// CHECK-LABEL:   tt.func @foo() {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i64
// CHECK:           func.call @proton_cpu_record_start(%[[CONSTANT_0]]) : (i64) -> ()
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i64
// CHECK:           func.call @proton_cpu_record_start(%[[CONSTANT_1]]) : (i64) -> ()
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : i64
// CHECK:           func.call @proton_cpu_record_end(%[[CONSTANT_2]]) : (i64) -> ()
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : i64
// CHECK:           func.call @proton_cpu_record_end(%[[CONSTANT_3]]) : (i64) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @foo() {
    proton.record start "kernel"
    proton.record start "load"
    proton.record end "load"
    proton.record end "kernel"
    tt.return
  }
}
