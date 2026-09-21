#pragma once

struct ErrorReport {
    double max_rel;
    double norm_rel;
};

// Frobenius-norm relative error and max (L-infinity) relative error,
// comparing a kernel's output C against the reference Cref.
ErrorReport compare_to_reference(int M, int N,
                                 const float* C, const float* Cref, int ldc);

// Verification threshold for fp32 GEMM with inner dimension K.
// Based on the standard rounding-error bound for summation: worst case
// grows like K*u, typical case like sqrt(K)*u, where u is fp32 unit
// roundoff (~1.19e-7). This sets the threshold comfortably above the
// typical case so correct kernels always pass, while still being tight
// enough to catch a real bug.
double verification_threshold(int K);