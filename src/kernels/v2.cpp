// v2 fixes the repeated memory traffic (keep the running total in a register the whole time) and started doing 4 numbers per instruction instead of 1.

#include "gemm.h"
#include <arm_neon.h>

// Fixes v1's remaining problem: v1 reads and writes C[i][j] from memory on every single k-step. Each row of the 4x4 tile now lives in one
// vector register for the whole k-loop and only touches memory once at the end. Each 4-wide load of B also serves all 4 rows, instead of
// being re-fetched per row as it effectively was before.
static void microkernel_4x4(int K,
                            const float* A, int lda,
                            const float* B, int ldb,
                            float* C, int ldc) {
    float32x4_t c0 = vdupq_n_f32(0.0f);
    float32x4_t c1 = vdupq_n_f32(0.0f);
    float32x4_t c2 = vdupq_n_f32(0.0f);
    float32x4_t c3 = vdupq_n_f32(0.0f);

    for (int p = 0; p < K; ++p) {
        float32x4_t b_vec = vld1q_f32(&B[p * ldb]);

        float32x4_t a0 = vdupq_n_f32(A[0 * lda + p]);
        c0 = vfmaq_f32(c0, a0, b_vec);
        float32x4_t a1 = vdupq_n_f32(A[1 * lda + p]);
        c1 = vfmaq_f32(c1, a1, b_vec);
        float32x4_t a2 = vdupq_n_f32(A[2 * lda + p]);
        c2 = vfmaq_f32(c2, a2, b_vec);
        float32x4_t a3 = vdupq_n_f32(A[3 * lda + p]);
        c3 = vfmaq_f32(c3, a3, b_vec);
    }

    vst1q_f32(&C[0 * ldc], c0);
    vst1q_f32(&C[1 * ldc], c1);
    vst1q_f32(&C[2 * ldc], c2);
    vst1q_f32(&C[3 * ldc], c3);
}

// Restriction: M and N must be multiples of 4 (no remainder handling yet). All sizes used so far (16, 64, 112, 240, 496, 1008) satisfy this.
void gemm_v2(int M, int N, int K,
            const float* A, int lda,
            const float* B, int ldb,
            float* C, int ldc) {
    for (int i = 0; i < M; i += 4) {
        for (int j = 0; j < N; j += 4) {
            microkernel_4x4(K,
                            &A[i * lda], lda,
                            &B[j], ldb,
                            &C[i * ldc + j], ldc);
        }
    }
}