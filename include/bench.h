// Header file to measure the execution time of a function.

#pragma once //if included more than once in same build, only process it once
#include <chrono>
#include <algorithm>

// Run fn several times, and then return the FASTEST time seen.
// Minimum because outside interference (OS, thermal) can only make a run slower than the machine's true capability, never faster.
template <typename F> // forwarding reference pattern, accept any callable type and F&& for effeciency
double time_min(F&& fn, int warmup = 3, int repeat = 7) {
    for (int i = 0; i < warmup; ++i) fn();
    double best = 1e300;
    for (int i = 0; i < repeat; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        double t = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
        best = std::min(best, t);
    }
    return best;
}