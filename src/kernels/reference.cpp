#include "gemm.h"

void gemm_reference(int M, int N, int K,
                    const float* A, int lda,
                    const float* B, int ldb,
                    float* C, int ldc) {
    for (int i = 0; i < M; ++i) { // loops over rows of A and C
        for (int j = 0; j < N; ++j) { // selects the j-th column of B and C
            double acc = 0.0;             // higher precision than the kernels being checked
            for (int p = 0; p < K; ++p) { //the dot product of the i-th row of A and the j-th column of B
                acc += (double)A[i * lda + p] * (double)B[p * ldb + j];
            }
            C[i * ldc + j] = (float)acc;  // cast down only at the very end
        }
    }
}