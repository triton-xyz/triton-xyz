// RUN: triton-xyz-opt --proton-to-xyz %s | FileCheck %s

module {
  // CHECK-DAG: func.func private @proton_cpu_record_start(i64)
  // CHECK-DAG: func.func private @proton_cpu_record_end(i64)
  // CHECK-LABEL: tt.func @foo
  // CHECK: %[[KERNEL_ID:.+]] = arith.constant 0 : i64
  // CHECK: func.call @proton_cpu_record_start(%[[KERNEL_ID]]) : (i64) -> ()
  // CHECK: %[[LOAD_ID:.+]] = arith.constant 1 : i64
  // CHECK: func.call @proton_cpu_record_start(%[[LOAD_ID]]) : (i64) -> ()
  // CHECK: %[[LOAD_END_ID:.+]] = arith.constant 1 : i64
  // CHECK: func.call @proton_cpu_record_end(%[[LOAD_END_ID]]) : (i64) -> ()
  // CHECK: %[[KERNEL_END_ID:.+]] = arith.constant 0 : i64
  // CHECK: func.call @proton_cpu_record_end(%[[KERNEL_END_ID]]) : (i64) -> ()
  tt.func @foo() {
    proton.record start "kernel"
    proton.record start "load"
    proton.record end "load"
    proton.record end "kernel"
    tt.return
  }
}
