#include "bench.h"
#include <arm_neon.h>
#include <cstdio>

// Sweeps through several values of CHAINS and ITERS to find the best performance on this machine.
// cat > /tmp/sweep.sh << 'EOF'
// #!/bin/bash
// for c in 4 8 12 16 20 24 32; do
//   for iters in 5000000 20000000; do
//     sed -i '' \
//       -e "s/constexpr int CHAINS = [0-9]*;/constexpr int CHAINS = $c;/" \
//       -e "s/constexpr long ITERS = [0-9]*L;/constexpr long ITERS = ${iters}L;/" \
//       src/bench/peak.cpp
//     clang++ -std=c++17 -O3 -mcpu=native -Iinclude src/bench/peak.cpp -o peak
//     echo -n "CHAINS=$c ITERS=$iters -> "; ./peak
//   done
// done
// EOF
// chmod +x /tmp/sweep.sh && /tmp/sweep.sh

int main() {
    constexpr int CHAINS = 20;      // several independent running totals
    constexpr long ITERS = 20000000L;

    // Each chain lives entirely in a register: no memory access in the loop.
    float32x4_t acc[CHAINS]; //4 packed 32-bit floats -> data type to let one CPU instruction operate on 4 floats at once
    for (int i = 0; i < CHAINS; ++i) acc[i] = vdupq_n_f32((float)i + 1.0f);
    const float32x4_t a = vdupq_n_f32(1.0000001f);
    const float32x4_t b = vdupq_n_f32(0.9999999f);

    double seconds = time_min([&]{
        for (long it = 0; it < ITERS; ++it)
            for (int i = 0; i < CHAINS; ++i)
                acc[i] = vfmaq_f32(acc[i], a, b);   // acc = acc*a + b
    });

    // Forces the compiler to keep the loop, otherwise it may notice the result is never used and delete the whole computation.
    volatile float sink = 0.0f;
    for (int i = 0; i < CHAINS; ++i) {
        float tmp[4]; vst1q_f32(tmp, acc[i]);
        for (int j = 0; j < 4; ++j) sink += tmp[j];
    }
    (void)sink;

    // Each FMA on a 4-lane vector does 4 multiplies + 4 adds = 8 FLOPs.
    double flops = (double)ITERS * CHAINS * 4.0 * 2.0;
    printf("measured peak: %.2f GFLOP/s\n", flops / seconds / 1e9);
    return 0;
}