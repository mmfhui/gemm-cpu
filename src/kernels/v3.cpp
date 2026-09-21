// Same 4x4 microkernel as v2. The optimization here is confining work to a block that fits in L1, so A/B/C tiles stay resident instead of being
// evicted and re-fetched as N grows past cache capacity.

#include "gemm.h"
#include <arm_neon.h>
#include <algorithm>

static void microkernel_4x4(int K,
                            const float* A, int lda,
                            const float* B, int ldb,
                            float* C, int ldc) {
    float32x4_t c0 = vld1q_f32(&C[0 * ldc]);
    float32x4_t c1 = vld1q_f32(&C[1 * ldc]);
    float32x4_t c2 = vld1q_f32(&C[2 * ldc]);
    float32x4_t c3 = vld1q_f32(&C[3 * ldc]);

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

constexpr int BLOCK = 32;   // predicted from L1d; to be swept

// Restriction: M, N, K must be multiples of 4 (microkernel requirement).
// BLOCK need not evenly divide M/N/K. The inner min() clamps the last partial block at each dimension's edge.
void gemm_v3(int M, int N, int K,
            const float* A, int lda,
            const float* B, int ldb,
            float* C, int ldc) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C[i * ldc + j] = 0.0f;

    for (int jc = 0; jc < N; jc += BLOCK) {
        int jb = std::min(BLOCK, N - jc);
        for (int pc = 0; pc < K; pc += BLOCK) {
            int pb = std::min(BLOCK, K - pc);
            for (int ic = 0; ic < M; ic += BLOCK) {
                int ib = std::min(BLOCK, M - ic);
                for (int i = 0; i < ib; i += 4) {
                    for (int j = 0; j < jb; j += 4) {
                        microkernel_4x4(pb,
                            &A[(ic+i) * lda + pc], lda,
                            &B[pc * ldb + (jc+j)], ldb,
                            &C[(ic+i) * ldc + (jc+j)], ldc);
                    }
                }
            }
        }
    }
}

// Sweep script to find the best BLOCK value for this machine:
// for b in 32 48 64 80 96 112 128; do
//   sed -i '' "s/constexpr int BLOCK = [0-9]*;/constexpr int BLOCK = $b;/" src/kernels/v3.cpp
//   clang++ -std=c++17 -O3 -mcpu=native -Iinclude src/bench/main.cpp \
//     src/kernels/v0.cpp src/kernels/v1.cpp src/kernels/v2.cpp src/kernels/v3.cpp \
//     src/kernels/reference.cpp src/bench/verify.cpp -o run_sweep
//   echo "=== BLOCK=$b ==="
//   ./run_sweep 2>&1 | awk '/=== v3 ===/{flag=1} flag'
// done