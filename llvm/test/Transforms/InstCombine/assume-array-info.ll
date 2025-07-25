; RUN: opt -S -passes='default<O1>,instcombine' < %s | FileCheck %s
; Test that llvm.assume calls with array_info operand bundles are preserved.
; through optimization passes, even at -O1 where regular assumes are removed.

declare void @llvm.assume(i1 noundef)

; Regular assume without operand bundles should be removed at -O1.
define void @test_regular_assume(ptr %p) {
; CHECK-LABEL: @test_regular_assume(
; CHECK-NOT: call void @llvm.assume
; CHECK: ret void
  call void @llvm.assume(i1 true)
  ret void
}

; Assume with array_info bundle should be preserved for loop optimizations.
define void @test_array_info_assume_preserved(ptr %arr) {
; CHECK-LABEL: @test_array_info_assume_preserved(
; CHECK: call void @llvm.assume(i1 true) [ "array_info"(ptr %arr, i64 2, i64 10, i64 20, i64 4) ]
; CHECK: ret void
  call void @llvm.assume(i1 true) [ "array_info"(ptr %arr, i64 2, i64 10, i64 20, i64 4) ]
  ret void
}

; Multiple operand bundles including array_info should preserve the assume.
define void @test_mixed_bundles_with_array_info(ptr %arr, ptr %p) {
; CHECK-LABEL: @test_mixed_bundles_with_array_info(
; CHECK: call void @llvm.assume(i1 true) [ "array_info"(ptr %arr, i64 1, i64 100, i64 4), "nonnull"(ptr %p) ]
; CHECK: ret void
  call void @llvm.assume(i1 true) [ "array_info"(ptr %arr, i64 1, i64 100, i64 4), "nonnull"(ptr %p) ]
  ret void
}
