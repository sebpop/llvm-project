; RUN: opt < %s -passes='print<delinearization>' -disable-output --use-gep-to-delinearize=false -debug-only=delinearize 2>&1 | FileCheck %s --check-prefix=CACHE
; REQUIRES: asserts

; This test verifies that the array_info cache works correctly by having
; multiple accesses to the same array. The cache should avoid repeated
; function entry block searches for the same base pointer.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@data = external global [4 x [8 x [16 x i32]]], align 16

; Verify that cache behavior is correct:
; - First access should miss cache and populate it.
; - Subsequent accesses should hit the cache.

; CACHE: Cache miss for instruction:   %val1 = load
; CACHE: Cache successful delinearization for instruction:   %val1 = load
; CACHE: Cache hit for instruction:   %val1 = load
; CACHE: Cache hit for instruction:   %val1 = load

; CACHE: Cache miss for instruction:   %val2 = load
; CACHE: Cache successful delinearization for instruction:   %val2 = load
; CACHE: Cache hit for instruction:   %val2 = load
; CACHE: Cache hit for instruction:   %val2 = load

; CACHE: Cache miss for instruction:   store i32 %sum
; CACHE: Cache successful delinearization for instruction:   store i32 %sum
; CACHE: Cache hit for instruction:   store i32 %sum
; CACHE: Cache hit for instruction:   store i32 %sum

define void @test_cache_effectiveness(i32 %n) {
entry:
  ; Add array_info metadata for the global array
  call void @llvm.assume(i1 true) ["array_info"(ptr @data, i64 3, i64 4, i64 8, i64 16, i64 4)]
  br label %for.i

for.i:
  %i = phi i32 [ 0, %entry ], [ %i.next, %for.i.inc ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i ], [ %j.next, %for.j.inc ]
  br label %for.k

for.k:
  %k = phi i32 [ 0, %for.j ], [ %k.next, %for.k.inc ]

  ; First access - should populate cache.
  %arrayidx1 = getelementptr inbounds [4 x [8 x [16 x i32]]], ptr @data, i32 0, i32 %i, i32 %j, i32 %k
  %val1 = load i32, ptr %arrayidx1, align 4

  ; Second access - should use cache (same base pointer @data.)
  %offset = add i32 %k, 1
  %arrayidx2 = getelementptr inbounds [4 x [8 x [16 x i32]]], ptr @data, i32 0, i32 %i, i32 %j, i32 %offset
  %val2 = load i32, ptr %arrayidx2, align 4

  ; Third access - should also use cache.
  %sum = add i32 %val1, %val2
  %arrayidx3 = getelementptr inbounds [4 x [8 x [16 x i32]]], ptr @data, i32 0, i32 %i, i32 %j, i32 %k
  store i32 %sum, ptr %arrayidx3, align 4
  br label %for.k.inc

for.k.inc:
  %k.next = add i32 %k, 1
  %k.cmp = icmp slt i32 %k.next, 16
  br i1 %k.cmp, label %for.k, label %for.j.inc

for.j.inc:
  %j.next = add i32 %j, 1
  %j.cmp = icmp slt i32 %j.next, 8
  br i1 %j.cmp, label %for.j, label %for.i.inc

for.i.inc:
  %i.next = add i32 %i, 1
  %i.cmp = icmp slt i32 %i.next, 4
  br i1 %i.cmp, label %for.i, label %exit

exit:
  ret void
}

declare void @llvm.assume(i1) #0

attributes #0 = { nounwind willreturn inaccessiblememonly }
