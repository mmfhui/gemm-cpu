// Same 4x4 microkernel as v3, but A is now read from a PACKED, contiguous scratch buffer instead of directly from the original strided matrix.
// The four A[row][p] reads per k-step become sequential reads from Apack instead of four separate jumps of lda elements each.
#include "gemm.h"
#include <arm_neon.h>
#include <algorithm>
#include <vector>

static void microkernel_4x4_packed(int K,
                                   const float* Apack,   // 4 rows, packed
                                   const float* B, int ldb,
                                   float* C, int ldc) {
    float32x4_t c0 = vld1q_f32(&C[0 * ldc]);
    float32x4_t c1 = vld1q_f32(&C[1 * ldc]);
    float32x4_t c2 = vld1q_f32(&C[2 * ldc]);
    float32x4_t c3 = vld1q_f32(&C[3 * ldc]);

    for (int p = 0; p < K; ++p) {
        float32x4_t b_vec = vld1q_f32(&B[p * ldb]);

        // Apack layout: for each p, the 4 values needed are stored contiguously as Apack[p*4 + row] (which is now sequential, not strided).
        float32x4_t a0 = vdupq_n_f32(Apack[p * 4 + 0]);
        c0 = vfmaq_f32(c0, a0, b_vec);
        float32x4_t a1 = vdupq_n_f32(Apack[p * 4 + 1]);
        c1 = vfmaq_f32(c1, a1, b_vec);
        float32x4_t a2 = vdupq_n_f32(Apack[p * 4 + 2]);
        c2 = vfmaq_f32(c2, a2, b_vec);
        float32x4_t a3 = vdupq_n_f32(Apack[p * 4 + 3]);
        c3 = vfmaq_f32(c3, a3, b_vec);
    }

    vst1q_f32(&C[0 * ldc], c0);
    vst1q_f32(&C[1 * ldc], c1);
    vst1q_f32(&C[2 * ldc], c2);
    vst1q_f32(&C[3 * ldc], c3);
}

// Copies 4 rows of A (each pb elements long, starting at column pc) into a contiguous scratch buffer, reordered so the microkernel's k-loop reads are sequential.
static void pack_A_4rows(const float* A, int lda, int pb, float* Apack) {
    for (int p = 0; p < pb; ++p)
        for (int row = 0; row < 4; ++row)
            Apack[p * 4 + row] = A[row * lda + p];
}

constexpr int BLOCK = 32;   // measured optimum from v3's sweep

void gemm_v4(int M, int N, int K,
            const float* A, int lda,
            const float* B, int ldb,
            float* C, int ldc) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C[i * ldc + j] = 0.0f;

    std::vector<float> Apack(BLOCK * 4);   // reused across the loop below

    for (int jc = 0; jc < N; jc += BLOCK) {
        int jb = std::min(BLOCK, N - jc);
        for (int pc = 0; pc < K; pc += BLOCK) {
            int pb = std::min(BLOCK, K - pc);
            for (int ic = 0; ic < M; ic += BLOCK) {
                int ib = std::min(BLOCK, M - ic);
                for (int i = 0; i < ib; i += 4) {
                    pack_A_4rows(&A[(ic+i) * lda + pc], lda, pb, Apack.data());
                    for (int j = 0; j < jb; j += 4) {
                        microkernel_4x4_packed(pb,
                            Apack.data(),
                            &B[pc * ldb + (jc+j)], ldb,
                            &C[(ic+i) * ldc + (jc+j)], ldc);
                    }
                }
            }
        }
    }
}