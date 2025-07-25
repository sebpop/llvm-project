//===---- Delinearization.cpp - MultiDimensional Index Delinearization ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements an analysis pass that tries to delinearize all GEP
// instructions in all loops using the SCEV analysis functionality. This pass is
// only used for testing purposes: if your pass needs delinearization, please
// use the on-demand SCEVAddRecExpr::delinearize() function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Delinearization.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionDivision.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DL_NAME "delinearize"
#define DEBUG_TYPE DL_NAME

static cl::opt<bool> UseFixedSizeArrayHeuristic(
    "delinearize-use-fixed-size-array-heuristic", cl::init(false), cl::Hidden,
    cl::desc("When printing analysis, use the heuristic for fixed-size arrays "
             "if the default delinearizetion fails."));

static cl::opt<bool> useGEPToDelinearize(
    "use-gep-to-delinearize", cl::init(true), cl::Hidden,
    cl::desc("validate both delinearization methods match."));

// Note: ArrayInfoCache was removed - use unified DelinearizationCache instead.

// Cache for delinearized subscripts to avoid redundant computation.
// Key: Instruction load/store/etc., Value: cached subscripts and
// sizes.

// Pretty printer implementation for DelinearizationCacheEntry.
void DelinearizationCacheEntry::print(raw_ostream &OS) const {
  if (!IsValid) {
    OS << "  [Invalid delinearization]";
    return;
  }

  OS << "  ArrayDecl";
  int NumSizes = Sizes.size();
  if (NumSizes > 0) {
    for (int i = 0; i < NumSizes - 1; i++)
      OS << "[" << *Sizes[i] << "]";
    // Print element size (last element in Sizes array).
    OS << " with elements of " << *Sizes[NumSizes - 1] << " bytes.\n";
  } else {
    OS << "[UnknownSize]\n";
  }

  OS << "  ArrayRef";
  for (const SCEV *S : Subscripts)
    OS << "[" << *S << "]";
  OS << "\n";
}

static DenseMap<Instruction *, DelinearizationCacheEntry> DelinearizationCache;

// Track the current function being analyzed for cache invalidation.
static const Function *CurrentCachedFunction = nullptr;

// Clear the cache when entering a new function context.
static void clearDelinearizationCache() {
  DelinearizationCache.clear();
  CurrentCachedFunction = nullptr;
}

// Check if we need to clear cache for function context switch.
static void checkAndClearCacheForFunction(const Function *F) {
  if (CurrentCachedFunction != F) {
    ::clearDelinearizationCache();
    CurrentCachedFunction = F;
    LLVM_DEBUG(dbgs() << "Switched to new function " << F->getName()
                      << ", cleared delinearization cache\n");
  }
}

// Public API wrappers for external access.
void llvm::clearDelinearizationCache() { ::clearDelinearizationCache(); }

void llvm::checkAndClearCacheForFunction(const Function *F) {
  ::checkAndClearCacheForFunction(F);
}

const DelinearizationCacheEntry *
llvm::getDelinearizationCacheEntry(Instruction *Inst) {
  auto CacheIt = DelinearizationCache.find(Inst);
  if (CacheIt != DelinearizationCache.end() && CacheIt->second.IsValid) {
    return &CacheIt->second;
  }
  return nullptr;
}

// Return true when S contains at least an undef value.
static inline bool containsUndefs(const SCEV *S) {
  return SCEVExprContains(S, [](const SCEV *S) {
    if (const auto *SU = dyn_cast<SCEVUnknown>(S))
      return isa<UndefValue>(SU->getValue());
    return false;
  });
}

namespace {

// Collect all steps of SCEV expressions.
struct SCEVCollectStrides {
  ScalarEvolution &SE;
  SmallVectorImpl<const SCEV *> &Strides;

  SCEVCollectStrides(ScalarEvolution &SE, SmallVectorImpl<const SCEV *> &S)
      : SE(SE), Strides(S) {}

  bool follow(const SCEV *S) {
    if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S))
      Strides.push_back(AR->getStepRecurrence(SE));
    return true;
  }

  bool isDone() const { return false; }
};

// Collect all SCEVUnknown and SCEVMulExpr expressions.
struct SCEVCollectTerms {
  SmallVectorImpl<const SCEV *> &Terms;

  SCEVCollectTerms(SmallVectorImpl<const SCEV *> &T) : Terms(T) {}

  bool follow(const SCEV *S) {
    if (isa<SCEVUnknown>(S) || isa<SCEVMulExpr>(S) ||
        isa<SCEVSignExtendExpr>(S)) {
      if (!containsUndefs(S))
        Terms.push_back(S);

      // Stop recursion: once we collected a term, do not walk its operands.
      return false;
    }

    // Keep looking.
    return true;
  }

  bool isDone() const { return false; }
};

// Check if a SCEV contains an AddRecExpr.
struct SCEVHasAddRec {
  bool &ContainsAddRec;

  SCEVHasAddRec(bool &ContainsAddRec) : ContainsAddRec(ContainsAddRec) {
    ContainsAddRec = false;
  }

  bool follow(const SCEV *S) {
    if (isa<SCEVAddRecExpr>(S)) {
      ContainsAddRec = true;

      // Stop recursion: once we collected a term, do not walk its operands.
      return false;
    }

    // Keep looking.
    return true;
  }

  bool isDone() const { return false; }
};

// Find factors that are multiplied with an expression that (possibly as a
// subexpression) contains an AddRecExpr. In the expression:
//
//  8 * (100 +  %p * %q * (%a + {0, +, 1}_loop))
//
// "%p * %q" are factors multiplied by the expression "(%a + {0, +, 1}_loop)"
// that contains the AddRec {0, +, 1}_loop. %p * %q are likely to be array size
// parameters as they form a product with an induction variable.
//
// This collector expects all array size parameters to be in the same MulExpr.
// It might be necessary to later add support for collecting parameters that are
// spread over different nested MulExpr.
struct SCEVCollectAddRecMultiplies {
  SmallVectorImpl<const SCEV *> &Terms;
  ScalarEvolution &SE;

  SCEVCollectAddRecMultiplies(SmallVectorImpl<const SCEV *> &T,
                              ScalarEvolution &SE)
      : Terms(T), SE(SE) {}

  bool follow(const SCEV *S) {
    if (auto *Mul = dyn_cast<SCEVMulExpr>(S)) {
      bool HasAddRec = false;
      SmallVector<const SCEV *, 0> Operands;
      for (const SCEV *Op : Mul->operands()) {
        const SCEVUnknown *Unknown = dyn_cast<SCEVUnknown>(Op);
        if (Unknown && !isa<CallInst>(Unknown->getValue())) {
          Operands.push_back(Op);
        } else if (Unknown) {
          HasAddRec = true;
        } else {
          bool ContainsAddRec = false;
          SCEVHasAddRec ContiansAddRec(ContainsAddRec);
          visitAll(Op, ContiansAddRec);
          HasAddRec |= ContainsAddRec;
        }
      }
      if (Operands.size() == 0)
        return true;

      if (!HasAddRec)
        return false;

      Terms.push_back(SE.getMulExpr(Operands));
      // Stop recursion: once we collected a term, do not walk its operands.
      return false;
    }

    // Keep looking.
    return true;
  }

  bool isDone() const { return false; }
};

} // end anonymous namespace

/// Find parametric terms in this SCEVAddRecExpr. We first for parameters in
/// two places:
///   1) The strides of AddRec expressions.
///   2) Unknowns that are multiplied with AddRec expressions.
void llvm::collectParametricTerms(ScalarEvolution &SE, const SCEV *Expr,
                                  SmallVectorImpl<const SCEV *> &Terms) {
  SmallVector<const SCEV *, 4> Strides;
  SCEVCollectStrides StrideCollector(SE, Strides);
  visitAll(Expr, StrideCollector);

  LLVM_DEBUG({
    dbgs() << "Strides:\n";
    for (const SCEV *S : Strides)
      dbgs() << "  " << *S << "\n";
  });

  for (const SCEV *S : Strides) {
    SCEVCollectTerms TermCollector(Terms);
    visitAll(S, TermCollector);
  }

  LLVM_DEBUG({
    dbgs() << "Terms:\n";
    for (const SCEV *T : Terms)
      dbgs() << "  " << *T << "\n";
  });

  SCEVCollectAddRecMultiplies MulCollector(Terms, SE);
  visitAll(Expr, MulCollector);
}

static bool findArrayDimensionsRec(ScalarEvolution &SE,
                                   SmallVectorImpl<const SCEV *> &Terms,
                                   SmallVectorImpl<const SCEV *> &Sizes) {
  int Last = Terms.size() - 1;
  const SCEV *Step = Terms[Last];

  // End of recursion.
  if (Last == 0) {
    if (const SCEVMulExpr *M = dyn_cast<SCEVMulExpr>(Step)) {
      SmallVector<const SCEV *, 2> Qs;
      for (const SCEV *Op : M->operands())
        if (!isa<SCEVConstant>(Op))
          Qs.push_back(Op);

      Step = SE.getMulExpr(Qs);
    }

    Sizes.push_back(Step);
    return true;
  }

  for (const SCEV *&Term : Terms) {
    // Normalize the terms before the next call to findArrayDimensionsRec.
    const SCEV *Q, *R;
    SCEVDivision::divide(SE, Term, Step, &Q, &R);

    // Bail out when GCD does not evenly divide one of the terms.
    if (!R->isZero())
      return false;

    Term = Q;
  }

  // Remove all SCEVConstants.
  erase_if(Terms, [](const SCEV *E) { return isa<SCEVConstant>(E); });

  if (Terms.size() > 0)
    if (!findArrayDimensionsRec(SE, Terms, Sizes))
      return false;

  Sizes.push_back(Step);
  return true;
}

// Returns true when one of the SCEVs of Terms contains a SCEVUnknown parameter.
static inline bool containsParameters(SmallVectorImpl<const SCEV *> &Terms) {
  for (const SCEV *T : Terms)
    if (SCEVExprContains(T, [](const SCEV *S) { return isa<SCEVUnknown>(S); }))
      return true;

  return false;
}

// Return the number of product terms in S.
static inline int numberOfTerms(const SCEV *S) {
  if (const SCEVMulExpr *Expr = dyn_cast<SCEVMulExpr>(S))
    return Expr->getNumOperands();
  return 1;
}

static const SCEV *removeConstantFactors(ScalarEvolution &SE, const SCEV *T) {
  if (isa<SCEVConstant>(T))
    return nullptr;

  if (isa<SCEVUnknown>(T))
    return T;

  if (const SCEVMulExpr *M = dyn_cast<SCEVMulExpr>(T)) {
    SmallVector<const SCEV *, 2> Factors;
    for (const SCEV *Op : M->operands())
      if (!isa<SCEVConstant>(Op))
        Factors.push_back(Op);

    return SE.getMulExpr(Factors);
  }

  return T;
}

void llvm::findArrayDimensions(ScalarEvolution &SE,
                               SmallVectorImpl<const SCEV *> &Terms,
                               SmallVectorImpl<const SCEV *> &Sizes,
                               const SCEV *ElementSize) {
  if (Terms.size() < 1 || !ElementSize)
    return;

  // Early return when Terms do not contain parameters: we do not delinearize
  // non parametric SCEVs.
  if (!containsParameters(Terms))
    return;

  LLVM_DEBUG({
    dbgs() << "Terms:\n";
    for (const SCEV *T : Terms)
      dbgs() << "  " << *T << "\n";
  });

  // Remove duplicates.
  array_pod_sort(Terms.begin(), Terms.end());
  Terms.erase(llvm::unique(Terms), Terms.end());

  // Put larger terms first.
  llvm::sort(Terms, [](const SCEV *LHS, const SCEV *RHS) {
    return numberOfTerms(LHS) > numberOfTerms(RHS);
  });

  // Try to divide all terms by the element size. If term is not divisible by
  // element size, proceed with the original term.
  for (const SCEV *&Term : Terms) {
    const SCEV *Q, *R;
    SCEVDivision::divide(SE, Term, ElementSize, &Q, &R);
    if (!Q->isZero())
      Term = Q;
  }

  SmallVector<const SCEV *, 4> NewTerms;

  // Remove constant factors.
  for (const SCEV *T : Terms)
    if (const SCEV *NewT = removeConstantFactors(SE, T))
      NewTerms.push_back(NewT);

  LLVM_DEBUG({
    dbgs() << "Terms after sorting:\n";
    for (const SCEV *T : NewTerms)
      dbgs() << "  " << *T << "\n";
  });

  if (NewTerms.empty() || !findArrayDimensionsRec(SE, NewTerms, Sizes)) {
    Sizes.clear();
    return;
  }

  // The last element to be pushed into Sizes is the size of an element.
  Sizes.push_back(ElementSize);

  LLVM_DEBUG({
    dbgs() << "Sizes:\n";
    for (const SCEV *S : Sizes)
      dbgs() << "  " << *S << "\n";
  });
}

void llvm::computeAccessFunctions(ScalarEvolution &SE, const SCEV *Expr,
                                  SmallVectorImpl<const SCEV *> &Subscripts,
                                  SmallVectorImpl<const SCEV *> &Sizes,
                                  Instruction *Inst) {
  // Early exit in case this SCEV is not an affine multivariate function.
  if (Sizes.empty())
    return;

  if (auto *AR = dyn_cast<SCEVAddRecExpr>(Expr))
    if (!AR->isAffine())
      return;

  // Check for function context switch and clear cache if needed.
  ::checkAndClearCacheForFunction(Inst->getFunction());

  // Check cache first using instruction as key.
  auto CacheIt = DelinearizationCache.find(Inst);
  if (CacheIt != DelinearizationCache.end() && CacheIt->second.IsValid) {
    LLVM_DEBUG({
      dbgs() << "Cache hit for instruction: " << *Inst << "\n";
      CacheIt->second.print(dbgs());
    });
    Subscripts.clear();
    Subscripts.append(CacheIt->second.Subscripts.begin(),
                      CacheIt->second.Subscripts.end());
    return;
  }

  LLVM_DEBUG(dbgs() << "Cache miss for instruction: " << *Inst << "\n");

  LLVM_DEBUG(dbgs() << "\ncomputeAccessFunctions for: " << *Inst << "\n"
                    << "Linearized Memory Access Function: " << *Expr << "\n");

  // Helper class to simplify SCEV expressions for delinearization.
  // Based on SCEVRemoveMax from Polly.
  class SCEVSimplifyForDelinearization final
      : public SCEVRewriteVisitor<SCEVSimplifyForDelinearization> {
  public:
    SCEVSimplifyForDelinearization(ScalarEvolution &SE)
        : SCEVRewriteVisitor(SE) {}

    static const SCEV *simplify(const SCEV *S, ScalarEvolution &SE) {
      SCEVSimplifyForDelinearization Simplifier(SE);
      const SCEV *OriginalS = S;
      S = Simplifier.visit(S);
      if (S != OriginalS)
        LLVM_DEBUG(dbgs() << "Simplified SCEV: " << *S << "\n");
      return S;
    }

    // Remove smax(0, expr) -> expr when expr >= 0 is implied by context.
    const SCEV *visitSMaxExpr(const SCEVSMaxExpr *Expr) {
      if (Expr->getNumOperands() == 2 && Expr->getOperand(0)->isZero()) {
        const SCEV *Inner = visit(Expr->getOperand(1));
        // If we can prove Inner >= 0, return Inner, otherwise keep smax.
        if (SE.isKnownNonNegative(Inner))
          return Inner;
      }
      return Expr;
    }
  };

  // Simplify the access function to handle max expressions.
  Expr = SCEVSimplifyForDelinearization::simplify(Expr, SE);

  // Helper function to normalize division to ensure access function stays
  // within array bounds. This solves the constraint: 0 <= AccessFunction <
  // UpperBound for all loop iterations.
  // Uses the same approach as AllIndicesInRange in DependenceAnalysis.cpp.
  auto normalizeDivisionForArrayBounds = [&SE](const SCEV *UpperBound,
                                               const SCEV *&OuterDimensions,
                                               const SCEV *&AccessFunction) {
    // Check if access function is already within bounds using SCEV analysis.

    // Access function must be non-negative.
    if (!SE.isKnownNonNegative(AccessFunction)) {
      LLVM_DEBUG(
          dbgs() << "Need normalization: access function may be negative: "
                 << *AccessFunction << "\n");
    } else {
      // Access function must be less than upper bound.
      if (auto *AccessType = dyn_cast<IntegerType>(AccessFunction->getType())) {
        if (auto *BoundType = dyn_cast<IntegerType>(UpperBound->getType())) {
          // Convert to common type if needed.
          const SCEV *TypeCompatibleBound = UpperBound;
          if (AccessType != BoundType) {
            unsigned CommonWidth =
                std::max(AccessType->getBitWidth(), BoundType->getBitWidth());
            Type *CommonType = IntegerType::get(SE.getContext(), CommonWidth);
            TypeCompatibleBound =
                SE.getTruncateOrZeroExtend(UpperBound, CommonType);
            // Also extend access function if needed.
            if (AccessFunction->getType() != CommonType) {
              AccessFunction =
                  SE.getTruncateOrZeroExtend(AccessFunction, CommonType);
            }
          }

          // Check if AccessFunction < UpperBound for all values.
          const SCEV *Diff =
              SE.getMinusSCEV(TypeCompatibleBound, AccessFunction);
          if (SE.isKnownPositive(Diff)) {
            LLVM_DEBUG(dbgs() << "Access function is within bounds, no "
                                 "normalization needed\n");
            return;
          }
        }
      }
      LLVM_DEBUG(dbgs() << "Need normalization: access function: "
                        << *AccessFunction
                        << " may overflow array's dimension upper bound: "
                        << *UpperBound << "\n");
    }

    // If we reach here, normalization is needed.
    auto *AccessAR = dyn_cast<SCEVAddRecExpr>(AccessFunction);
    // Can only normalize AddRec expressions (even after simplification).
    if (!AccessAR || !AccessAR->isAffine()) {
      LLVM_DEBUG(dbgs() << "Cannot normalize non-affine access function\n");
      return;
    }

    const SCEV *AccessStart = AccessAR->getStart();
    const SCEV *AccessStep = AccessAR->getStepRecurrence(SE);
    const Loop *L = AccessAR->getLoop();

    LLVM_DEBUG(dbgs() << "  Normalizing access function: " << *AccessFunction
                      << " to fit within upper bound: " << *UpperBound << "\n");

    // For now, implement a conservative approach: only normalize when we can
    // determine the access function violates bounds AND we can safely fix it.

    // This requires both the access function and upper bound to be simple
    // enough.
    auto *BoundConst = dyn_cast<SCEVConstant>(UpperBound);
    auto *StepConst = dyn_cast<SCEVConstant>(AccessStep);
    auto *StartConst = dyn_cast<SCEVConstant>(AccessStart);

    if (!BoundConst || !StepConst || !StartConst) {
      LLVM_DEBUG(dbgs() << "  Cannot normalize: non-constant components\n");
      return;
    }

    const APInt &BoundValue = BoundConst->getAPInt();
    const APInt &StepValue = StepConst->getAPInt();
    const APInt &StartValue = StartConst->getAPInt();

    // Only handle reasonably sized integers to avoid overflow.
    if (BoundValue.getBitWidth() > 64 || StepValue.getBitWidth() > 64 ||
        StartValue.getBitWidth() > 64) {
      LLVM_DEBUG(dbgs() << "  Cannot normalize: bit width too large\n");
      return;
    }

    // Check if normalization is actually needed.
    // For a simple case: if step >= bound, we need normalization.
    if (StepValue.ult(BoundValue) && StartValue.isNonNegative() &&
        StartValue.ult(BoundValue)) {
      LLVM_DEBUG(
          dbgs()
          << "  No normalization needed: access appears to be in bounds\n");
      return;
    }

    // Simple normalization: only proceed if step >= bound.
    if (!StepValue.uge(BoundValue)) {
      LLVM_DEBUG(dbgs() << "  No normalization needed: step < bound\n");
      return;
    }

    LLVM_DEBUG(dbgs() << "  Applying step normalization\n");

    uint64_t StepVal = StepValue.getLimitedValue();
    uint64_t BoundVal = BoundValue.getLimitedValue();

    uint64_t Adjustment = StepVal / BoundVal;
    int64_t NewStepVal = (int64_t)StepVal - (int64_t)(Adjustment * BoundVal);

    LLVM_DEBUG(dbgs() << "  Step normalization: " << StepVal << " -> "
                      << NewStepVal << " (adjustment: " << Adjustment << ")\n");

    // Create new SCEVs.
    Type *AccessType = AccessStep->getType();
    const SCEV *StepAdjustment = SE.getConstant(AccessType, Adjustment);
    const SCEV *NewAccessStep = SE.getConstant(AccessType, NewStepVal, true);

    // Update outer dimensions.
    if (auto *OuterAR = dyn_cast<SCEVAddRecExpr>(OuterDimensions)) {
      const SCEV *OuterStart = OuterAR->getStart();
      const SCEV *OuterStep = OuterAR->getStepRecurrence(SE);
      const SCEV *NewOuterStep = SE.getAddExpr(OuterStep, StepAdjustment);
      OuterDimensions = SE.getAddRecExpr(OuterStart, NewOuterStep, L,
                                         OuterAR->getNoWrapFlags());
    } else {
      OuterDimensions = SE.getAddRecExpr(OuterDimensions, StepAdjustment, L,
                                         SCEV::FlagAnyWrap);
    }

    // Update access function.
    AccessFunction = SE.getAddRecExpr(AccessStart, NewAccessStep, L,
                                      AccessAR->getNoWrapFlags());

    LLVM_DEBUG(dbgs() << "  Normalized outer dimensions: " << *OuterDimensions
                      << "\n");
    LLVM_DEBUG(dbgs() << "  Normalized access function: " << *AccessFunction
                      << "\n");
  };

  const SCEV *Res = Expr;
  int Last = Sizes.size() - 1;

  for (int i = Last; i >= 0; i--) {
    const SCEV *Q, *R;
    const SCEV *Size = SCEVSimplifyForDelinearization::simplify(Sizes[i], SE);

    SCEVDivision::divide(SE, Res, Size, &Q, &R);

    LLVM_DEBUG({
      dbgs() << "Computing 'MemAccFn / Sizes[" << i << "]':\n";
      dbgs() << "  MemAccFn: " << *Res << "\n";
      dbgs() << "  Sizes[" << i << "]: " << *Size << "\n";
      dbgs() << "  Quotient (Leftover): " << *Q << "\n";
      dbgs() << "  Remainder (Subscript Access Function): " << *R << "\n";
    });

    // Normalize to ensure the subscript access function (aka. remainder R in
    // the division above) stays within the array subscript bounds [0, size).
    normalizeDivisionForArrayBounds(Size, Q, R);

    Res = Q;

    // Do not record the last subscript corresponding to the size of elements in
    // the array.
    if (i == Last) {

      // Bail out if the byte offset is non-zero.
      if (!R->isZero()) {
        Subscripts.clear();
        Sizes.clear();
        return;
      }

      continue;
    }

    // Record the access function for the current subscript.
    LLVM_DEBUG(dbgs() << "Subscripts push_back Remainder: " << *R << "\n");
    Subscripts.push_back(R);
  }

  // Also push in last position the quotient "Res = Q" of the last division: it
  // will be the access function of the outermost array dimension.
  if (!Res->isZero()) {
    // This is only needed when the outermost array size is not known.  Res = 0
    // when the outermost array dimension is known, as for example when reading
    // array sizes from array_info.
    Subscripts.push_back(Res);
    LLVM_DEBUG(dbgs() << "Subscripts push_back Res: " << *Res << "\n");
  }

  std::reverse(Subscripts.begin(), Subscripts.end());

  LLVM_DEBUG({
    dbgs() << "Subscripts:\n";
    for (const SCEV *S : Subscripts)
      dbgs() << "  " << *S << "\n";
    dbgs() << "\n";
  });

  // Cache the result at the end.
  if (!Subscripts.empty()) {
    DelinearizationCache[Inst] = DelinearizationCacheEntry(Subscripts, Sizes);
    LLVM_DEBUG({
      dbgs() << "Cache successful delinearization for instruction: " << *Inst
             << "\n";
      DelinearizationCache[Inst].print(dbgs());
    });

  } else {
    DelinearizationCache[Inst] = DelinearizationCacheEntry();
    LLVM_DEBUG({
      dbgs() << "Cache negative delinearization result for instruction: "
             << *Inst << "\n";
      DelinearizationCache[Inst].print(dbgs());
    });
  }
}

/// Splits the SCEV into two vectors of SCEVs representing the subscripts and
/// sizes of an array access. Returns the remainder of the delinearization that
/// is the offset start of the array.  The SCEV->delinearize algorithm computes
/// the multiples of SCEV coefficients: that is a pattern matching of sub
/// expressions in the stride and base of a SCEV corresponding to the
/// computation of a GCD (greatest common divisor) of base and stride.  When
/// SCEV->delinearize fails, it returns the SCEV unchanged.
///
/// For example: when analyzing the memory access A[i][j][k] in this loop nest
///
///  void foo(long n, long m, long o, double A[n][m][o]) {
///
///    for (long i = 0; i < n; i++)
///      for (long j = 0; j < m; j++)
///        for (long k = 0; k < o; k++)
///          A[i][j][k] = 1.0;
///  }
///
/// the delinearization input is the following AddRec SCEV:
///
///  AddRec: {{{%A,+,(8 * %m * %o)}<%for.i>,+,(8 * %o)}<%for.j>,+,8}<%for.k>
///
/// From this SCEV, we are able to say that the base offset of the access is %A
/// because it appears as an offset that does not divide any of the strides in
/// the loops:
///
///  CHECK: Base offset: %A
///
/// and then SCEV->delinearize determines the size of some of the dimensions of
/// the array as these are the multiples by which the strides are happening:
///
///  CHECK: ArrayDecl[UnknownSize][%m][%o] with elements of sizeof(double)
///  bytes.
///
/// Note that the outermost dimension remains of UnknownSize because there are
/// no strides that would help identifying the size of the last dimension: when
/// the array has been statically allocated, one could compute the size of that
/// dimension by dividing the overall size of the array by the size of the known
/// dimensions: %m * %o * 8.
///
/// Finally delinearize provides the access functions for the array reference
/// that does correspond to A[i][j][k] of the above C testcase:
///
///  CHECK: ArrayRef[{0,+,1}<%for.i>][{0,+,1}<%for.j>][{0,+,1}<%for.k>]
///
/// The testcases are checking the output of a function pass:
/// DelinearizationPass that walks through all loads and stores of a function
/// asking for the SCEV of the memory access with respect to all enclosing
/// loops, calling SCEV->delinearize on that and printing the results.
void llvm::delinearize(ScalarEvolution &SE, const SCEV *Expr,
                       SmallVectorImpl<const SCEV *> &Subscripts,
                       SmallVectorImpl<const SCEV *> &Sizes,
                       const SCEV *ElementSize, Instruction *Inst) {
  // Check for function context switch and clear cache if needed.
  ::checkAndClearCacheForFunction(Inst->getFunction());

  // Check cache first to avoid expensive parametric term collection.
  auto CacheIt = DelinearizationCache.find(Inst);
  if (CacheIt != DelinearizationCache.end() && CacheIt->second.IsValid) {
    LLVM_DEBUG({
      dbgs() << "Cache hit for instruction: " << *Inst << "\n";
      CacheIt->second.print(dbgs());
    });

    Subscripts.clear();
    Subscripts.append(CacheIt->second.Subscripts.begin(),
                      CacheIt->second.Subscripts.end());
    Sizes.clear();
    Sizes.append(CacheIt->second.Sizes.begin(), CacheIt->second.Sizes.end());
    return;
  }

  LLVM_DEBUG(dbgs() << "Cache miss for instruction: " << *Inst << "\n");

  // Clear output vectors.
  Subscripts.clear();
  Sizes.clear();

  // First step: collect parametric terms.
  SmallVector<const SCEV *, 4> Terms;
  collectParametricTerms(SE, Expr, Terms);

  if (Terms.empty())
    return;

  // Second step: find subscript sizes.
  findArrayDimensions(SE, Terms, Sizes, ElementSize);

  if (Sizes.empty())
    return;

  // Third step: compute the access functions for each subscript.
  computeAccessFunctions(SE, Expr, Subscripts, Sizes, Inst);
}

static std::optional<APInt> tryIntoAPInt(const SCEV *S) {
  if (const auto *Const = dyn_cast<SCEVConstant>(S))
    return Const->getAPInt();
  return std::nullopt;
}

/// Collects the absolute values of constant steps for all induction variables.
/// Returns true if we can prove that all step recurrences are constants and \p
/// Expr is divisible by \p ElementSize. Each step recurrence is stored in \p
/// Steps after divided by \p ElementSize.
static bool collectConstantAbsSteps(ScalarEvolution &SE, const SCEV *Expr,
                                    SmallVectorImpl<uint64_t> &Steps,
                                    uint64_t ElementSize) {
  // End of recursion. The constant value also must be a multiple of
  // ElementSize.
  if (const auto *Const = dyn_cast<SCEVConstant>(Expr)) {
    const uint64_t Mod = Const->getAPInt().urem(ElementSize);
    return Mod == 0;
  }

  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Expr);
  if (!AR || !AR->isAffine())
    return false;

  const SCEV *Step = AR->getStepRecurrence(SE);
  std::optional<APInt> StepAPInt = tryIntoAPInt(Step);
  if (!StepAPInt)
    return false;

  APInt Q;
  uint64_t R;
  APInt::udivrem(StepAPInt->abs(), ElementSize, Q, R);
  if (R != 0)
    return false;

  // Bail out when the step is too large.
  std::optional<uint64_t> StepVal = Q.tryZExtValue();
  if (!StepVal)
    return false;

  Steps.push_back(*StepVal);
  return collectConstantAbsSteps(SE, AR->getStart(), Steps, ElementSize);
}

bool llvm::findFixedSizeArrayDimensions(ScalarEvolution &SE, const SCEV *Expr,
                                        SmallVectorImpl<uint64_t> &Sizes,
                                        const SCEV *ElementSize) {
  if (!ElementSize)
    return false;

  std::optional<APInt> ElementSizeAPInt = tryIntoAPInt(ElementSize);
  if (!ElementSizeAPInt || *ElementSizeAPInt == 0)
    return false;

  std::optional<uint64_t> ElementSizeConst = ElementSizeAPInt->tryZExtValue();

  // Early exit when ElementSize is not a positive constant.
  if (!ElementSizeConst)
    return false;

  if (!collectConstantAbsSteps(SE, Expr, Sizes, *ElementSizeConst) ||
      Sizes.empty()) {
    Sizes.clear();
    return false;
  }

  // At this point, Sizes contains the absolute step recurrences for all
  // induction variables. Each step recurrence must be a multiple of the size of
  // the array element. Assuming that the each value represents the size of an
  // array for each dimension, attempts to restore the length of each dimension
  // by dividing the step recurrence by the next smaller value. For example, if
  // we have the following AddRec SCEV:
  //
  //   AddRec: {{{0,+,2048}<%for.i>,+,256}<%for.j>,+,8}<%for.k> (ElementSize=8)
  //
  // Then Sizes will become [256, 32, 1] after sorted. We don't know the size of
  // the outermost dimension, the next dimension will be computed as 256 / 32 =
  // 8, and the last dimension will be computed as 32 / 1 = 32. Thus it results
  // in like Arr[UnknownSize][8][32] with elements of size 8 bytes, where Arr is
  // a base pointer.
  //
  // TODO: Catch more cases, e.g., when a step recurrence is not divisible by
  // the next smaller one, like A[i][3*j].
  llvm::sort(Sizes.rbegin(), Sizes.rend());
  Sizes.erase(llvm::unique(Sizes), Sizes.end());

  // The last element in Sizes should be ElementSize. At this point, all values
  // in Sizes are assumed to be divided by ElementSize, so replace it with 1.
  assert(Sizes.back() != 0 && "Unexpected zero size in Sizes.");
  Sizes.back() = 1;

  for (unsigned I = 0; I + 1 < Sizes.size(); I++) {
    uint64_t PrevSize = Sizes[I + 1];
    if (Sizes[I] % PrevSize) {
      Sizes.clear();
      return false;
    }
    Sizes[I] /= PrevSize;
  }

  // Finally, the last element in Sizes should be ElementSize.
  Sizes.back() = *ElementSizeConst;
  return true;
}

/// Splits the SCEV into two vectors of SCEVs representing the subscripts and
/// sizes of an array access, assuming that the array is a fixed size array.
///
/// E.g., if we have the code like as follows:
///
///  double A[42][8][32];
///  for i
///    for j
///      for k
///        use A[i][j][k]
///
/// The access function will be represented as an AddRec SCEV like:
///
///  AddRec: {{{0,+,2048}<%for.i>,+,256}<%for.j>,+,8}<%for.k> (ElementSize=8)
///
/// Then findFixedSizeArrayDimensions infers the size of each dimension of the
/// array based on the fact that the value of the step recurrence is a multiple
/// of the size of the corresponding array element. In the above example, it
/// results in the following:
///
///  CHECK: ArrayDecl[UnknownSize][8][32] with elements of 8 bytes.
///
/// Finally each subscript will be computed as follows:
///
///  CHECK: ArrayRef[{0,+,1}<%for.i>][{0,+,1}<%for.j>][{0,+,1}<%for.k>]
///
/// Note that this function doesn't check the range of possible values for each
/// subscript, so the caller should perform additional boundary checks if
/// necessary.
///
/// Also note that this function doesn't guarantee that the original array size
/// is restored "correctly". For example, in the following case:
///
///  double A[42][4][64];
///  double B[42][8][32];
///  for i
///    for j
///      for k
///        use A[i][j][k]
///        use B[i][2*j][k]
///
/// The access function for both accesses will be the same:
///
///  AddRec: {{{0,+,2048}<%for.i>,+,512}<%for.j>,+,8}<%for.k> (ElementSize=8)
///
/// The array sizes for both A and B will be computed as
/// ArrayDecl[UnknownSize][4][64], which matches for A, but not for B.
///
/// TODO: At the moment, this function can handle only simple cases. For
/// example, we cannot handle a case where a step recurrence is not divisible
/// by the next smaller step recurrence, e.g., A[i][3*j].
bool llvm::delinearizeFixedSizeArray(ScalarEvolution &SE, const SCEV *Expr,
                                     SmallVectorImpl<const SCEV *> &Subscripts,
                                     SmallVectorImpl<const SCEV *> &Sizes,
                                     const SCEV *ElementSize,
                                     Instruction *Inst) {
  // Check for function context switch and clear cache if needed.
  ::checkAndClearCacheForFunction(Inst->getFunction());

  // Check cache first to avoid expensive array dimension finding.
  auto CacheIt = DelinearizationCache.find(Inst);
  if (CacheIt != DelinearizationCache.end() && CacheIt->second.IsValid) {
    LLVM_DEBUG({
      dbgs() << "Cache hit for instruction: " << *Inst << "\n";
      CacheIt->second.print(dbgs());
    });

    Subscripts.clear();
    Subscripts.append(CacheIt->second.Subscripts.begin(),
                      CacheIt->second.Subscripts.end());
    Sizes.clear();
    Sizes.append(CacheIt->second.Sizes.begin(), CacheIt->second.Sizes.end());
    return !Subscripts.empty();
  }

  LLVM_DEBUG(dbgs() << "Cache miss for instruction: " << *Inst << "\n");

  // Clear output vectors.
  Subscripts.clear();
  Sizes.clear();

  // First step: find the fixed array size.
  SmallVector<uint64_t, 4> ConstSizes;
  if (!findFixedSizeArrayDimensions(SE, Expr, ConstSizes, ElementSize)) {
    Sizes.clear();
    return false;
  }

  // Convert the constant size to SCEV.
  for (uint64_t Size : ConstSizes)
    Sizes.push_back(SE.getConstant(Expr->getType(), Size));

  // Second step: compute the access functions for each subscript.
  computeAccessFunctions(SE, Expr, Subscripts, Sizes, Inst);

  return !Subscripts.empty();
}

bool llvm::getIndexExpressionsFromGEP(ScalarEvolution &SE,
                                      const GetElementPtrInst *GEP,
                                      SmallVectorImpl<const SCEV *> &Subscripts,
                                      SmallVectorImpl<int> &Sizes) {
  assert(Subscripts.empty() && Sizes.empty() &&
         "Expected output lists to be empty on entry to this function.");
  assert(GEP && "getIndexExpressionsFromGEP called with a null GEP");
  LLVM_DEBUG(dbgs() << "\nGEP to delinearize: " << *GEP << "\n");
  Type *Ty = nullptr;
  bool DroppedFirstDim = false;
  for (unsigned i = 1; i < GEP->getNumOperands(); i++) {
    const SCEV *Expr = SE.getSCEV(GEP->getOperand(i));
    if (i == 1) {
      Ty = GEP->getSourceElementType();
      if (auto *Const = dyn_cast<SCEVConstant>(Expr))
        if (Const->getValue()->isZero()) {
          DroppedFirstDim = true;
          continue;
        }
      Subscripts.push_back(Expr);
      LLVM_DEBUG(dbgs() << "Subscripts push_back: " << *Expr << "\n");
      continue;
    }

    auto *ArrayTy = dyn_cast<ArrayType>(Ty);
    if (!ArrayTy) {
      LLVM_DEBUG(dbgs() << "GEP delinearize failed: " << Ty
                        << " is not an array type.\n");
      Subscripts.clear();
      Sizes.clear();
      return false;
    }

    Subscripts.push_back(Expr);
    LLVM_DEBUG(dbgs() << "Subscripts push_back: " << *Expr << "\n");
    if (!(DroppedFirstDim && i == 2))
      Sizes.push_back(ArrayTy->getNumElements());

    Ty = ArrayTy->getElementType();
  }
  LLVM_DEBUG({
    dbgs() << "Subscripts:\n";
    for (const SCEV *S : Subscripts)
      dbgs() << *S << "\n";
    dbgs() << "\n";
  });

  return !Subscripts.empty();
}

static bool delinearizeUsingArrayInfo(ScalarEvolution *SE, Instruction *Inst,
                                      GetElementPtrInst *SrcGEP,
                                      const SCEV *AccessFn,
                                      SmallVectorImpl<const SCEV *> &Subscripts,
                                      SmallVectorImpl<int> &Sizes) {
  // Check for function context switch and clear cache if needed.
  ::checkAndClearCacheForFunction(Inst->getFunction());

  // Check cache first to avoid expensive array_info search.
  auto CacheIt = DelinearizationCache.find(Inst);
  if (CacheIt != DelinearizationCache.end() && CacheIt->second.IsValid) {
    LLVM_DEBUG({
      dbgs() << "Cache hit for instruction: " << *Inst << "\n";
      CacheIt->second.print(dbgs());
    });

    // Convert cached SCEV subscripts to output format.
    Subscripts.clear();
    Subscripts.append(CacheIt->second.Subscripts.begin(),
                      CacheIt->second.Subscripts.end());

    // Convert cached SCEV sizes to int sizes for compatibility.
    Sizes.clear();
    for (const SCEV *S : CacheIt->second.Sizes) {
      if (auto *Const = dyn_cast<SCEVConstant>(S)) {
        const APInt &APVal = Const->getAPInt();
        if (APVal.isSignedIntN(32)) {
          int intValue = APVal.getSExtValue();
          Sizes.push_back(intValue);
        }
      }
    }
    return !Sizes.empty();
  }

  LLVM_DEBUG(dbgs() << "Cache miss for instruction: " << *Inst << "\n");

  const SCEVUnknown *BasePointer =
      dyn_cast<SCEVUnknown>(SE->getPointerBase(AccessFn));
  if (!BasePointer)
    return false;

  Value *BasePtr = BasePointer->getValue();
  SmallVector<const SCEV *, 4> SCEVSizes;

  if (!tryGetArrayInfoFromAssumes(*SE, BasePtr, Inst, SCEVSizes))
    return false;

  // Get the full SCEV expression and subtract the base pointer to get
  // offset-only expression.
  const SCEV *FullExpr = SE->getSCEV(SrcGEP);
  const SCEV *Expr = SE->getMinusSCEV(FullExpr, BasePointer);

  computeAccessFunctions(*SE, Expr, Subscripts, SCEVSizes, Inst);
  if (SCEVSizes.empty() || Subscripts.empty())
    return false;

  // TODO: Remove the following code. Convert SCEV sizes to int sizes. This
  // conversion is only needed as long as getIndexExpressionsFromGEP is still
  // around. Remove this code and change the interface of
  // tryDelinearizeFixedSizeImpl to take a SmallVectorImpl<const SCEV *> &Sizes.
  for (const SCEV *S : SCEVSizes) {
    if (auto *Const = dyn_cast<SCEVConstant>(S)) {
      const APInt &APVal = Const->getAPInt();
      if (APVal.isSignedIntN(32)) {
        int intValue = APVal.getSExtValue();
        Sizes.push_back(intValue);
      }
    }
  }

  return !Sizes.empty();
}

bool llvm::tryDelinearizeFixedSizeImpl(
    ScalarEvolution *SE, Instruction *Inst, const SCEV *AccessFn,
    SmallVectorImpl<const SCEV *> &Subscripts, SmallVectorImpl<int> &Sizes) {
  // Check for function context switch and clear cache if needed.
  ::checkAndClearCacheForFunction(Inst->getFunction());

  // Check cache first to avoid expensive delinearization work.
  auto CacheIt = DelinearizationCache.find(Inst);
  if (CacheIt != DelinearizationCache.end() && CacheIt->second.IsValid) {
    LLVM_DEBUG({
      dbgs() << "Cache hit for instruction: " << *Inst << "\n";
      CacheIt->second.print(dbgs());
    });

    Subscripts.clear();
    Subscripts.append(CacheIt->second.Subscripts.begin(),
                      CacheIt->second.Subscripts.end());

    // Convert cached SCEV sizes to int sizes for compatibility.
    Sizes.clear();
    for (const SCEV *S : CacheIt->second.Sizes) {
      if (auto *Const = dyn_cast<SCEVConstant>(S)) {
        const APInt &APVal = Const->getAPInt();
        if (APVal.isSignedIntN(32)) {
          int intValue = APVal.getSExtValue();
          Sizes.push_back(intValue);
        }
      }
    }
    return !Sizes.empty();
  }

  LLVM_DEBUG(dbgs() << "Cache miss for instruction: " << *Inst << "\n");

  Value *SrcPtr = getLoadStorePointerOperand(Inst);

  // Check the simple case where the array dimensions are fixed size.
  auto *SrcGEP = dyn_cast<GetElementPtrInst>(SrcPtr);
  if (!SrcGEP)
    return false;

  // When flag useGEPToDelinearize is false, delinearize only using array_info.
  if (!useGEPToDelinearize)
    return delinearizeUsingArrayInfo(SE, Inst, SrcGEP, AccessFn, Subscripts,
                                     Sizes);

  // TODO: Remove all the following code once we are satisfied with array_info.
  // Run both methods when useGEPToDelinearize is true: validation is enabled.

  // Store results from both methods.
  SmallVector<const SCEV *, 4> GEPSubscripts, ArrayInfoSubscripts;
  SmallVector<int, 4> GEPSizes, ArrayInfoSizes;

  // GEP-based delinearization.
  bool GEPSuccess =
      getIndexExpressionsFromGEP(*SE, SrcGEP, GEPSubscripts, GEPSizes);

  // Array_info delinearization.
  bool ArrayInfoSuccess = delinearizeUsingArrayInfo(
      SE, Inst, SrcGEP, AccessFn, ArrayInfoSubscripts, ArrayInfoSizes);

  // Validate consistency between methods.
  if (GEPSuccess && ArrayInfoSuccess) {
    // If both methods succeeded, validate they produce the same results.
    // Compare sizes arrays.
    if (GEPSizes.size() + 2 != ArrayInfoSizes.size()) {
      LLVM_DEBUG({
        dbgs() << "WARN: Size arrays have different lengths!\n";
        dbgs() << "GEP sizes count: " << GEPSizes.size() << "\n"
               << "ArrayInfo sizes count: " << ArrayInfoSizes.size() << "\n";
      });
    }

    for (size_t i = 0; i < GEPSizes.size(); ++i) {
      if (GEPSizes[i] != ArrayInfoSizes[i + 1]) {
        LLVM_DEBUG({
          dbgs() << "WARN: Size arrays differ at index " << i << "!\n";
          dbgs() << "GEP size[" << i << "]: " << GEPSizes[i] << "\n"
                 << "ArrayInfo size[" << i + 1 << "]: " << ArrayInfoSizes[i + 1]
                 << "\n";
        });
      }
    }

    // Compare subscripts arrays.
    if (GEPSubscripts.size() != ArrayInfoSubscripts.size()) {
      LLVM_DEBUG({
        dbgs() << "WARN: Subscript arrays have different lengths!\n";
        dbgs() << "  GEP subscripts count: " << GEPSubscripts.size() << "\n"
               << "  ArrayInfo subscripts count: " << ArrayInfoSubscripts.size()
               << "\n";

        dbgs() << "  GEP subscripts:\n";
        for (size_t i = 0; i < GEPSubscripts.size(); ++i)
          dbgs() << "    subscript[" << i << "]: " << *GEPSubscripts[i] << "\n";

        dbgs() << "  ArrayInfo subscripts:\n";
        for (size_t i = 0; i < ArrayInfoSubscripts.size(); ++i)
          dbgs() << "    subscript[" << i << "]: " << *ArrayInfoSubscripts[i]
                 << "\n";
      });
    }

    for (size_t i = 0; i < GEPSubscripts.size(); ++i) {
      const SCEV *GEPS = GEPSubscripts[i];
      const SCEV *AIS = ArrayInfoSubscripts[i];
      // FIXME: there's no good way to compare two scevs: don't abort, warn.
      if (GEPS != AIS || !SE->getMinusSCEV(GEPS, AIS)->isZero()) {
        LLVM_DEBUG({
          dbgs() << "WARN: Subscript arrays differ at index " << i << "!\n";
          dbgs() << "  GEP subscript[" << i << "]: " << *GEPSubscripts[i]
                 << "\n"
                 << "  ArrayInfo subscript[" << i
                 << "]: " << *ArrayInfoSubscripts[i] << "\n";
        });
      }
    }

    LLVM_DEBUG(dbgs() << "SUCCESS: Both delinearization methods produced "
                         "identical results\n");
  } else if (GEPSuccess && !ArrayInfoSuccess) {
    LLVM_DEBUG({
      dbgs() << "WARNING: array_info failed and GEP analysis succeeded.\n";
      dbgs() << "  Instruction: " << *Inst << "\n";
      dbgs() << "  Using GEP analysis results despite array_info failure\n";
    });
  } else if (!GEPSuccess && ArrayInfoSuccess) {
    LLVM_DEBUG({
      dbgs() << "WARNING: GEP failed and array_info analysis succeeded.\n";
      dbgs() << "  Instruction: " << *Inst << "\n";
      dbgs() << "  Using array_info analysis results despite GEP failure\n";
    });
  } else if (!GEPSuccess && !ArrayInfoSuccess) {
    LLVM_DEBUG({
      dbgs() << "WARNING: both GEP and array_info analysis failed.\n";
      dbgs() << "  Instruction: " << *Inst << "\n";
    });
  }

  // Choose which result to use.
  // Prefer array_info when available.
  if (ArrayInfoSuccess) {
    Subscripts = std::move(ArrayInfoSubscripts);
    Sizes = std::move(ArrayInfoSizes);
    return true;
  }

  // Both failed.
  if (!GEPSuccess)
    return false;

  // Return GEP-based delinearization.
  Subscripts = std::move(GEPSubscripts);
  Sizes = std::move(GEPSizes);

  // Check that the two size arrays are non-empty and equal in length and
  // value.
  // TODO: it would be better to let the caller to clear Subscripts, similar
  // to how we handle Sizes.
  if (Sizes.empty() || Subscripts.size() <= 1) {
    Subscripts.clear();
    return false;
  }

  // Check that for identical base pointers we do not miss index offsets
  // that have been added before this GEP is applied.
  Value *SrcBasePtr = SrcGEP->getOperand(0)->stripPointerCasts();
  const SCEVUnknown *SrcBase =
      dyn_cast<SCEVUnknown>(SE->getPointerBase(AccessFn));
  if (!SrcBase || SrcBasePtr != SrcBase->getValue()) {
    Subscripts.clear();
    return false;
  }

  assert(Subscripts.size() == Sizes.size() + 1 &&
         "Expected equal number of entries in the list of size and "
         "subscript.");

  return true;
}

// Extract dimensions and element size from an array_info operand bundle and
// convert them to SCEV. Returns true on success, false on failure.
static bool extractFromBundle(ScalarEvolution &SE,
                              const OperandBundleUse &Bundle, uint64_t Rank,
                              const Instruction *CtxI, Value *BasePtr,
                              SmallVectorImpl<const SCEV *> &Sizes) {
  // Extract dimensions.
  Sizes.clear();
  Type *I64Ty = Type::getInt64Ty(CtxI->getContext());
  for (uint64_t i = 0; i < Rank; ++i) {
    Value *DimVal = Bundle.Inputs[2 + i];
    if (auto *DimConst = dyn_cast<ConstantInt>(DimVal)) {
      const SCEV *DimSCEV = SE.getConstant(I64Ty, DimConst->getZExtValue());
      Sizes.push_back(DimSCEV);
    } else {
      // Handle non-constant dimensions by creating SCEV from value.
      const SCEV *DimSCEV = SE.getSCEV(DimVal);
      if (DimSCEV && !isa<SCEVCouldNotCompute>(DimSCEV)) {
        Sizes.push_back(DimSCEV);
      } else {
        LLVM_DEBUG(dbgs() << "Failed to create SCEV for dimension value\n");
        Sizes.clear();
        return false;
      }
    }
  }

  // Extract element_size.
  Value *DimVal = Bundle.Inputs[2 + Rank];
  if (auto *DimConst = dyn_cast<ConstantInt>(DimVal)) {
    const SCEV *DimSCEV = SE.getConstant(I64Ty, DimConst->getZExtValue());
    if (SE.getElementSize(const_cast<Instruction *>(CtxI)) != DimSCEV) {
      LLVM_DEBUG(dbgs() << "  element_size != getElementSize(CtxI)\n");
      Sizes.clear();
      return false;
    }
    Sizes.push_back(DimSCEV);
  } else {
    LLVM_DEBUG(dbgs() << "  element_size is not constant\n");
    Sizes.clear();
    return false;
  }

  LLVM_DEBUG({
    dbgs() << "Found array_info for base pointer " << *BasePtr << "\n";
    dbgs() << "Rank: " << Rank << "\n";
    dbgs() << "Dimensions: ";
    for (const SCEV *Size : Sizes)
      dbgs() << *Size << " ";
    dbgs() << "\n";
  });

  return true;
}

bool llvm::tryGetArrayInfoFromAssumes(ScalarEvolution &SE, Value *BasePtr,
                                      const Instruction *CtxI,
                                      SmallVectorImpl<const SCEV *> &Sizes) {
  LLVM_DEBUG(
      dbgs()
      << "tryGetArrayInfoFromAssumes: Searching array_info from instruction: "
      << *CtxI << "\n");

  // Search only in the function entry block since array_info assumes are
  // typically placed there.
  const Function *F = CtxI->getFunction();
  const BasicBlock *EntryBB = &F->getEntryBlock();

  OperandBundleUse Bundle;
  uint64_t Rank = 0;

  // Search the entry block for array_info assume intrinsics.  If array_info is
  // not found in entry BB, do not search elsewhere to avoid expensive IR walks.
  // Inline pass or other passes that may move the assume stmts should be fixed.
  for (const Instruction &I : *EntryBB) {
    auto *Assume = dyn_cast<IntrinsicInst>(&I);
    if (!Assume)
      continue;

    if (Assume->getIntrinsicID() != Intrinsic::assume)
      continue;

    LLVM_DEBUG(dbgs() << "Found assume: " << *Assume << "\n");

    // Check if this assume has an array_info operand bundle.
    auto OptBundle = Assume->getOperandBundle("array_info");
    if (!OptBundle)
      continue;

    Bundle = *OptBundle;

    // Check if the base pointer matches.
    if (Bundle.Inputs.size() < 2)
      continue;

    Value *AssumeBasePtr = Bundle.Inputs[0];
    // Strip casts to compare base pointers.
    if (AssumeBasePtr->stripPointerCasts() != BasePtr->stripPointerCasts())
      continue;

    // Extract rank.
    auto *RankConst = dyn_cast<ConstantInt>(Bundle.Inputs[1]);
    if (!RankConst)
      continue;
    Rank = RankConst->getZExtValue();

    // Verify we have the right number of operands: ptr + rank + rank*dim
    // + element_size.
    if (Bundle.Inputs.size() != 2 + Rank + 1)
      continue;

    return extractFromBundle(SE, Bundle, Rank, CtxI, BasePtr, Sizes);
  }

  LLVM_DEBUG(dbgs() << "tryGetArrayInfoFromAssumes: No array_info found "
                       "in function entry block\n");
  return false;
}

namespace {

void printDelinearization(raw_ostream &O, Function *F, LoopInfo *LI,
                          ScalarEvolution *SE) {
  O << "Printing analysis 'Delinearization' for function '" << F->getName()
    << "':";
  for (Instruction &Inst : instructions(F)) {
    // Only analyze loads and stores.
    if (!isa<StoreInst>(&Inst) && !isa<LoadInst>(&Inst) &&
        !isa<GetElementPtrInst>(&Inst))
      continue;

    const BasicBlock *BB = Inst.getParent();
    // Delinearize the memory access as analyzed in all the surrounding loops.
    // Do not analyze memory accesses outside loops.
    for (Loop *L = LI->getLoopFor(BB); L != nullptr; L = L->getParentLoop()) {
      const SCEV *OriginalAccessFn =
          SE->getSCEVAtScope(getPointerOperand(&Inst), L);

      const SCEVUnknown *BasePointer =
          dyn_cast<SCEVUnknown>(SE->getPointerBase(OriginalAccessFn));
      // Do not delinearize if we cannot find the base pointer.
      if (!BasePointer)
        break;

      const SCEV *AccessFn = SE->getMinusSCEV(OriginalAccessFn, BasePointer);

      O << "\n";
      O << "Inst:" << Inst << "\n";
      O << "In Loop with Header: " << L->getHeader()->getName() << "\n";
      O << "AccessFunction: " << *AccessFn << "\n";

      SmallVector<const SCEV *, 3> Subscripts, Sizes;

      auto IsDelinearizationFailed = [&]() {
        return Subscripts.size() == 0 || Sizes.size() == 0;
      };

      delinearize(*SE, AccessFn, Subscripts, Sizes, SE->getElementSize(&Inst),
                  &Inst);

      // Store classic delinearization results for comparison.
      SmallVector<const SCEV *, 3> ClassicSubscripts = Subscripts;
      SmallVector<const SCEV *, 3> ClassicSizes = Sizes;

      // Always try array_info delinearization for load/store instructions when
      // array_info metadata is available, as it often provides more complete
      // results.
      bool SucceededWithArrayInfo = false;
      if (isa<LoadInst>(&Inst) || isa<StoreInst>(&Inst)) {
        SmallVector<int, 4> IntSizes;
        SmallVector<const SCEV *, 3> ArrayInfoSubscripts;
        // Use original AccessFn for array_info delinearization as it will
        // handle base pointer subtraction internally.
        if (tryDelinearizeFixedSizeImpl(SE, &Inst, OriginalAccessFn,
                                        ArrayInfoSubscripts, IntSizes)) {
          LLVM_DEBUG({
            dbgs() << "tryDelinearizeFixedSizeImpl succeeded with "
                   << ArrayInfoSubscripts.size() << " subscripts and "
                   << IntSizes.size() << " sizes: ";
            for (int size : IntSizes)
              dbgs() << size << " ";
            dbgs() << "\n";
          });
          // Convert int sizes to SCEV sizes for consistent output.
          SmallVector<const SCEV *, 3> ArrayInfoSizes;
          for (int Size : IntSizes) {
            ArrayInfoSizes.push_back(
                SE->getConstant(Type::getInt64Ty(Inst.getContext()), Size));
          }

          // Prefer array_info results when available as they typically
          // provide more complete dimension information.
          Subscripts = std::move(ArrayInfoSubscripts);
          Sizes = std::move(ArrayInfoSizes);
          SucceededWithArrayInfo = true;
          LLVM_DEBUG({
            dbgs() << "Using array_info results: " << Subscripts.size()
                   << " subscripts, " << Sizes.size() << " sizes\n";
          });
        } else {
          LLVM_DEBUG(dbgs() << "tryDelinearizeFixedSizeImpl failed\n");
        }
      }

      // If array_info didn't work, fall back to classic results or error out.
      if (!SucceededWithArrayInfo) {
        Subscripts = std::move(ClassicSubscripts);
        Sizes = std::move(ClassicSizes);
      }

      if (UseFixedSizeArrayHeuristic && IsDelinearizationFailed()) {
        Subscripts.clear();
        Sizes.clear();
        delinearizeFixedSizeArray(*SE, AccessFn, Subscripts, Sizes,
                                  SE->getElementSize(&Inst), &Inst);
      }
      if (IsDelinearizationFailed()) {
        O << "failed to delinearize\n";
        continue;
      }

      O << "Base offset: " << *BasePointer << "\n";
      O << "ArrayDecl";
      int NumSubscripts = Subscripts.size();
      int NumSizes = Sizes.size();

      // Handle different size relationships between Subscripts and Sizes.
      if (NumSizes > 0) {
        // Print array dimensions (all but the last size, which is element
        // size).
        for (int i = 0; i < NumSizes - 1; i++)
          O << "[" << *Sizes[i] << "]";

        // Print element size (last element in Sizes array).
        O << " with elements of " << *Sizes[NumSizes - 1] << " bytes.\n";
      } else {
        O << " unknown sizes.\n";
      }

      O << "ArrayRef";
      for (int i = 0; i < NumSubscripts; i++)
        O << "[" << *Subscripts[i] << "]";
      O << "\n";
    }
  }
}

} // end anonymous namespace

DelinearizationPrinterPass::DelinearizationPrinterPass(raw_ostream &OS)
    : OS(OS) {}
PreservedAnalyses DelinearizationPrinterPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  // Clear the delinearization cache when starting analysis of a new function.
  ::clearDelinearizationCache();

  printDelinearization(OS, &F, &AM.getResult<LoopAnalysis>(F),
                       &AM.getResult<ScalarEvolutionAnalysis>(F));

  return PreservedAnalyses::all();
}
