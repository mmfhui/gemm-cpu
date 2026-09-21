#include "gemm.h"
#include "bench.h"
#include "verify.h"
#include <vector>
#include <random>
#include <cstdio>
#include <cstring>

struct Kernel {
    const char* name;
    void (*fn)(int, int, int, const float*, int, const float*, int, float*, int);
};

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    int sizes[] = {16, 64, 112, 240, 496, 1008};
    double peak = 143.88;   // measured ceiling from compute.cpp

    Kernel kernels[] = {
        {"v0", gemm_v0},
        {"v1", gemm_v1},
        {"v2", gemm_v2},
        {"v3", gemm_v3},
        {"v4", gemm_v4},
        {"v5", gemm_v5},
    };

    for (auto& k : kernels) {
        printf("=== %s ===\n", k.name);
        printf("%-6s %10s %8s %14s %14s %12s %s\n",
               "N", "GFLOP/s", "%peak", "max_rel", "norm_rel", "threshold", "status");

        for (int n : sizes) {
            std::vector<float> A(n*n), B(n*n), C(n*n), Cref(n*n);
            for (auto& x : A) x = dist(rng);
            for (auto& x : B) x = dist(rng);

            gemm_reference(n, n, n, A.data(), n, B.data(), n, Cref.data(), n);

            double seconds = time_min([&]{
                k.fn(n, n, n, A.data(), n, B.data(), n, C.data(), n);
            });

            double flops = 2.0 * n * n * n;
            double gflops = flops / seconds / 1e9;

            ErrorReport err = compare_to_reference(n, n, C.data(), Cref.data(), n);
            double thresh = verification_threshold(n);
            const char* status = (err.norm_rel < thresh) ? "PASS" : "FAIL";

            printf("%-6d %10.3f %7.2f%% %14.3e %14.3e %10.3e %s\n",
                   n, gflops, 100.0*gflops/peak, err.max_rel, err.norm_rel, thresh, status);
        }
        printf("\n");
    }
    return 0;
}