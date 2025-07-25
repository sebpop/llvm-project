// Check that LLVM IR contains the array_info assumes.
// RUN: %clang_cc1 -emit-llvm -O2 %s -o - | FileCheck %s --check-prefix=LLVM-IR
// LLVM-IR: call void @llvm.assume(i1 true) [ "array_info"(ptr %arr, i64 2, i64 10, i64 20, i64 4) ]
// LLVM-IR: call void @llvm.assume(i1 true) [ "array_info"(ptr %arr, i64 3, i64 5, i64 10, i64 15, i64 4) ]

// Test array delinearization with 2d and 3d loop nests.
// RUN: %clang_cc1 -emit-llvm -O2 %s -o - | opt -passes='print<delinearization>' -disable-output 2>&1 | FileCheck %s --check-prefix=DELINEARIZE

// DELINEARIZE: Printing analysis 'Delinearization' for function 'test_2d_loop_access':
// DELINEARIZE: Base offset: %arr
// DELINEARIZE: ArrayDecl[10][20] with elements of 4 bytes.
// DELINEARIZE: ArrayRef[{{.*}}][{{.*}}]

// DELINEARIZE: Printing analysis 'Delinearization' for function 'test_3d_loop_access':
// DELINEARIZE: Base offset: %arr
// DELINEARIZE: ArrayDecl[5][10][15] with elements of 4 bytes.
// DELINEARIZE: ArrayRef[{{.*}}][{{.*}}][{{.*}}]

extern volatile int sink;
extern void external_use(void* ptr);

void test_2d_loop_access(int n, int m) {
    int arr[10][20];

    for (int i = 0; i < n && i < 10; i++) {
        for (int j = 0; j < m && j < 20; j++) {
            arr[i][j] = i * 20 + j;
            sink = arr[i][j];
        }
    }
    external_use(arr);
}

void test_3d_loop_access(int n, int m, int p) {
    int arr[5][10][15];

    for (int i = 0; i < n && i < 5; i++) {
        for (int j = 0; j < m && j < 10; j++) {
            for (int k = 0; k < p && k < 15; k++) {
                arr[i][j][k] = i * 150 + j * 15 + k;
                sink = arr[i][j][k];
            }
        }
    }
    external_use(arr);
}
