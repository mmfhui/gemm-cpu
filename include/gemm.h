#pragma once

//   Every GEMM kernel in the project shares this signature:
//   C = A * B
//
//   M, N, K: matrix dimensions (C is MxN, A is MxK, B is KxN).
//   lda, ldb, ldc: the actual row length in memory of A, B, C respectively.
//   Keeping them separate from M/N/K matters for tiling and padding, where the storage stride can differ from the logical dimension.

void gemm_reference(int M, int N, int K,
                    const float* A, int lda,
                    const float* B, int ldb,
                    float* C, int ldc);

void gemm_v0(int M, int N, int K,
            const float* A, int lda,
            const float* B, int ldb,
            float* C, int ldc);

void gemm_v1(int M, int N, int K,
            const float* A, int lda,
            const float* B, int ldb,
            float* C, int ldc);