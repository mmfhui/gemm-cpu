// The triple loop implementation of GEMM with the inner two loops swapped for cache locality.
#include "gemm.h"

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