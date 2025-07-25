; RUN: opt -S -passes=instcombine < %s | FileCheck %s
; Test that llvm.assume calls with array_info operand bundles are preserved.
; in realistic scenarios where the array operations can't be constant-folded.

declare void @llvm.assume(i1 noundef)
declare void @external_function(ptr)
declare i32 @get_dynamic_value()

define void @realistic_array_usage() {
; CHECK-LABEL: @realistic_array_usage(
; CHECK: %arr = alloca [10 x [20 x i32]]
; CHECK: call void @llvm.assume(i1 true) [ "array_info"(ptr %arr, i64 2, i64 10, i64 20, i64 4) ]
; CHECK: call void @external_function(ptr {{.*}} %arr)
; CHECK: ret void
  %arr = alloca [10 x [20 x i32]], align 4
  call void @llvm.assume(i1 true) [ "array_info"(ptr %arr, i64 2, i64 10, i64 20, i64 4) ]

  ; Pass array to external function - can't be optimized away.
  call void @external_function(ptr %arr)
  ret void
}

; Dynamic array access that can't be constant-folded.
define i32 @dynamic_array_access() {
; CHECK-LABEL: @dynamic_array_access(
; CHECK: %arr = alloca [100 x i32]
; CHECK: call void @llvm.assume(i1 true) [ "array_info"(ptr %arr, i64 1, i64 100, i64 4) ]
; CHECK: ret i32
  %arr = alloca [100 x i32], align 4
  call void @llvm.assume(i1 true) [ "array_info"(ptr %arr, i64 1, i64 100, i64 4) ]

  ; Initialize array with some value.
  %store_gep = getelementptr inbounds i32, ptr %arr, i32 0
  store i32 42, ptr %store_gep, align 4

  ; Dynamic index from external function.
  %idx = call i32 @get_dynamic_value()
  %gep = getelementptr inbounds i32, ptr %arr, i32 %idx
  %val = load i32, ptr %gep, align 4
  ret i32 %val
}
