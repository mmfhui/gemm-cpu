#include "verify.h"
#include <cmath>

ErrorReport compare_to_reference(int M, int N,
                                 const float* C, const float* Cref, int ldc) {
    double max_rel = 0.0;
    double sq_diff_sum = 0.0;
    double sq_ref_sum = 0.0;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double c    = C[i * ldc + j];
            double cref = Cref[i * ldc + j];
            double diff = std::fabs(c - cref);

            if (std::fabs(cref) > 1e-30) {
                double rel = diff / std::fabs(cref);
                if (rel > max_rel) max_rel = rel;
            }

            sq_diff_sum += diff * diff;
            sq_ref_sum  += cref * cref;
        }
    }

    ErrorReport r;
    r.max_rel  = max_rel;
    r.norm_rel = std::sqrt(sq_diff_sum) / std::sqrt(sq_ref_sum);
    return r;
}

double verification_threshold(int K) {
    const double u = 5.9604645e-8;   // 2^-24, fp32 unit roundoff (not eps=2^-23)
    double typical = std::sqrt((double)K) * u;
    return 20.0 * typical;                    // comfortably above typical,
                                               // still well below worst-case K*u
}
