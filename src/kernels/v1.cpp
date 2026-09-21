// The triple loop implementation of GEMM with the inner two loops swapped for cache locality.
#include "gemm.h"

// clang++ -std=c++17 -O3 -mcpu=native -Iinclude src/bench/main.cpp \
//   src/kernels/v0.cpp src/kernels/v1.cpp src/kernels/reference.cpp \
//   src/bench/verify.cpp -o run_v0v1
// ./run_v0v1

void gemm_v1(int M, int N, int K,
            const float* A, int lda,
            const float* B, int ldb,
            float* C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) C[i * ldc + j] = 0.0f;
        for (int p = 0; p < K; ++p) {
            float a = A[i * lda + p];
            for (int j = 0; j < N; ++j) {
                C[i * ldc + j] += a * B[p * ldb + j];
            }
        }
    }
}