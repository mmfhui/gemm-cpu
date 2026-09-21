// 8x8 tile: 16 independent accumulator vectors (8 rows x 2 vectors of 4 lanes), versus v4's 4x4 tile (only 4). Motivation: compute.cpp's
// own peak-throughput sweep found this chip needs ~20 independent FMA chains to approach its measured ceiling; v4 only had 4.
// Swept against 8x4, 4x8, and 16x4. 8x8 won at every size tested, by 5-15 points over the next-best shape. Shape balance appears to matter more
// than raw register-count efficiency, likely by giving the compiler's scheduler more independent work to interleave across both dimensions rather
// than one long row-wise dependency chain.

#include "gemm.h"
#include <arm_neon.h>
#include <algorithm>
#include <vector>

static void microkernel_8x8_packed(int K,
                                   const float* Apack,   // 8 rows, packed
                                   const float* B, int ldb,
                                   float* C, int ldc) {
    float32x4_t c[8][2];
    for (int r = 0; r < 8; ++r) {
        c[r][0] = vld1q_f32(&C[r * ldc + 0]);
        c[r][1] = vld1q_f32(&C[r * ldc + 4]);
    }

    for (int p = 0; p < K; ++p) {
        float32x4_t b0 = vld1q_f32(&B[p * ldb + 0]);
        float32x4_t b1 = vld1q_f32(&B[p * ldb + 4]);

        for (int r = 0; r < 8; ++r) {
            float32x4_t a = vdupq_n_f32(Apack[p * 8 + r]);
            c[r][0] = vfmaq_f32(c[r][0], a, b0);
            c[r][1] = vfmaq_f32(c[r][1], a, b1);
        }
    }

    for (int r = 0; r < 8; ++r) {
        vst1q_f32(&C[r * ldc + 0], c[r][0]);
        vst1q_f32(&C[r * ldc + 4], c[r][1]);
    }
}

static void pack_A_8rows(const float* A, int lda, int pb, float* Apack) {
    for (int p = 0; p < pb; ++p)
        for (int row = 0; row < 8; ++row)
            Apack[p * 8 + row] = A[row * lda + p];
}

constexpr int BLOCK = 32;   // unchanged from v3/v4; must be a multiple
                            // of 8 now (32 qualifies)

// Restriction: M and N must be multiples of 8 (tile size), up from 4.
// All benchmark sizes (16,64,112,240,496,1008) satisfy this.
void gemm_v5(int M, int N, int K,
            const float* A, int lda,
            const float* B, int ldb,
            float* C, int ldc) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C[i * ldc + j] = 0.0f;

    std::vector<float> Apack(BLOCK * 8);

    for (int jc = 0; jc < N; jc += BLOCK) {
        int jb = std::min(BLOCK, N - jc);
        for (int pc = 0; pc < K; pc += BLOCK) {
            int pb = std::min(BLOCK, K - pc);
            for (int ic = 0; ic < M; ic += BLOCK) {
                int ib = std::min(BLOCK, M - ic);
                for (int i = 0; i < ib; i += 8) {
                    pack_A_8rows(&A[(ic+i) * lda + pc], lda, pb, Apack.data());
                    for (int j = 0; j < jb; j += 8) {
                        microkernel_8x8_packed(pb,
                            Apack.data(),
                            &B[pc * ldb + (jc+j)], ldb,
                            &C[(ic+i) * ldc + (jc+j)], ldc);
                    }
                }
            }
        }
    }
}
