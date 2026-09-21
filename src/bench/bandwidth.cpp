// later check if removing the write does anything
#include "bench.h"
#include <vector>
#include <cstdio>
#include <cstdlib>

// sweep through several values of n to find the best performance on this machine.
// clang++ -std=c++17 -O3 -mcpu=native -Iinclude src/bench/bandwidth.cpp -o bandwidth
// for n in 1000000 4000000 16000000 64000000 128000000; do
// ./bandwidth $n
// done
// clang++ -std=c++17 -O3 -mcpu=native -Iinclude src/bench/bandwidth.cpp -o bandwidth
// for n in 1000000 4000000 16000000 64000000 128000000; do
// ./bandwidth $n
// done 

int main(int argc, char** argv) {
    size_t n = (argc > 1) ? std::strtoul(argv[1], nullptr, 10) : (1u << 24);

    std::vector<float> a(n), b(n), c(n);
    for (size_t i = 0; i < n; ++i) { b[i] = 1.0f; c[i] = 2.0f; }
    const float s = 3.0f;

    double seconds = time_min([&]{
        for (size_t i = 0; i < n; ++i)
            a[i] = b[i] + s * c[i];
    });

    // Force every element of 'a' to actually be computed, after timing
    // ends -- this is a full read, not just three spot-checks, because a
    // clever compiler could otherwise prove only a few elements matter.
    double checksum = 0.0;
    for (size_t i = 0; i < n; ++i) checksum += a[i];
    volatile double sink = checksum;
    (void)sink;

    double bytes = 3.0 * n * sizeof(float);   // b read, c read, a written
    printf("n=%zu (%.1f MB/array) -> %.2f GB/s\n",
           n, n * sizeof(float) / 1e6, bytes / seconds / 1e9);
    return 0;
}