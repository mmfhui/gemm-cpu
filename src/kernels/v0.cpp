// The naive triple loop implementation of GEMM
#include "gemm.h"

// clang++ -std=c++17 -O3 -mcpu=native -Iinclude src/bench/main.cpp \
//   src/kernels/v0.cpp src/kernels/reference.cpp src/bench/verify.cpp -o run_v0
// ./run_v0

void gemm_v0(int M, int N, int K,
            const float* A, int lda,
            const float* B, int ldb,
            float* C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int p = 0; p < K; ++p) {
                acc += A[i * lda + p] * B[p * ldb + j];
            }
            C[i * ldc + j] = acc;
        }
    }
}


