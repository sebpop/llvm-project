; RUN: opt < %s -disable-output "-passes=print<da>" -aa-pipeline=basic-aa 2>&1 \
; RUN: | FileCheck %s

; Check that dependence analysis correctly handles flip-flop of base addresses.
; Bug 41488 - https://github.com/llvm/llvm-project/issues/41488

; CHECK-LABEL: bug41488_test1
; CHECK-NOT: da analyze - none!

define float @bug41488_test1() {
entry:
  %g = alloca float, align 4
  %h = alloca float, align 4
  br label %for.body

for.body:
  %p = phi float* [ %g, %entry ], [ %q, %for.body ]
  %q = phi float* [ %h, %entry ], [ %p, %for.body ]
  %0 = load float, float* %p, align 4
  store float undef, float* %q, align 4
  %branch_cond = fcmp ugt float %0, 0.0
  br i1 %branch_cond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret float undef
}


; CHECK-LABEL: bug41488_test2
; CHECK-NOT: da analyze - none!

define void @bug41488_test2(i32 %n) {
entry:
  %g = alloca float, align 4
  %h = alloca float, align 4
  br label %for.body

for.body:
  %i = phi i32 [0, %entry ], [ %inc, %for.body ]
  %p = phi float* [ %g, %entry ], [ %q, %for.body ]
  %q = phi float* [ %h, %entry ], [ %p, %for.body ]
  %0 = load float, float* %p, align 4
  store float 0.0, float* %q, align 4
  %inc = add nuw i32 %i, 1
  %branch_cond = icmp ult i32 %i, %n
  br i1 %branch_cond, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}

; Bug 53942 - https://github.com/llvm/llvm-project/issues/53942
; CHECK-LABEL: bug53942_foo
; CHECK-NOT: da analyze - none!

define void @bug53942_foo(i32 noundef %n, ptr noalias nocapture noundef writeonly %A, ptr noalias nocapture noundef %B) {
entry:
  %cmp8 = icmp sgt i32 %n, 1
  br i1 %cmp8, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext nneg i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 1, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %ptr1.011 = phi ptr [ %A, %for.body.preheader ], [ %ptr2.09, %for.body ]
  %ptr2.09 = phi ptr [ %B, %for.body.preheader ], [ %ptr1.011, %for.body ]
  %.pre = load double, ptr %B, align 8
  %arrayidx2 = getelementptr inbounds double, ptr %ptr1.011, i64 %indvars.iv
  store double %.pre, ptr %arrayidx2, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
