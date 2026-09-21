# gemm-cpu

Single-threaded fp32 matrix multiply (`C = A * B`) on CPU, built up one optimization at a time from a naive triple loop to a NEON microkernel running at **~80% of measured hardware peak** on an Apple M4 Pro.

The point of the repo is the progression, not just the final kernel: every version is kept, benchmarked against the same harness, and verified against a reference implementation. [docs/REPORT.md](docs/REPORT.md) is the full log — machine characterization, the roofline analysis, a prediction before each version, and the measured result after.

## Results

Measured ceiling on my machine: **143.88 GFLOP/s** (see [Characterization](#characterization)).
Percentages are of that ceiling; all sizes pass verification.

| Kernel | What changed                             | N=112  | N=1008 |
|--------|------------------------------------------|--------|--------|
| v0     | naive triple loop (ijk)                  |  2.90% |  1.95% |
| v1     | loop reorder (ikj) for unit-stride access| 27.51% | 23.64% |
| v2     | 4x4 register tile, NEON intrinsics       | 38.14% | 25.59% |
| v3     | cache blocking (BLOCK=32)                | 49.86% | 45.02% |
| v4     | pack A into a contiguous buffer          | 51.79% | 45.26% |
| v5     | widened 8x8 microkernel (16 accumulators)| 80.39% | 72.47% |

Each step was triggered by a specific diagnosed bottleneck in the step before it. Full per-size tables, including error metrics, are in the report.

## Building and running

No build system yet — everything is a single `clang++` invocation.

```sh
# all kernels, all sizes, with verification
clang++ -std=c++17 -O3 -mcpu=native -Iinclude \
    src/bench/main.cpp src/bench/verify.cpp src/kernels/*.cpp -o run_all
./run_all
```

The two characterization microbenchmarks build the same way and are run on their own:

```sh
clang++ -std=c++17 -O3 -mcpu=native -Iinclude src/bench/peak.cpp -o peak
./peak                     # sustained FMA throughput -> compute ceiling

clang++ -std=c++17 -O3 -mcpu=native -Iinclude src/bench/bandwidth.cpp -o bandwidth
./bandwidth 64000000       # STREAM Triad -> memory bandwidth (array size in elements)
```

Both files carry the sweep scripts used to find their plateaus in a comment at the top.

## Layout

| Path                     | Contents                                                        |
|--------------------------|-----------------------------------------------------------------|
| [include/gemm.h](include/gemm.h)       | the one signature every kernel shares            |
| [include/bench.h](include/bench.h)     | `time_min` — warmup, repeat, keep the fastest run |
| [include/verify.h](include/verify.h)   | error metrics and the size-dependent pass threshold |
| [src/kernels/](src/kernels/)           | `reference.cpp` plus `v0`–`v5`                   |
| [src/bench/main.cpp](src/bench/main.cpp) | the benchmark + verification driver            |
| [src/bench/peak.cpp](src/bench/peak.cpp), [bandwidth.cpp](src/bench/bandwidth.cpp) | machine characterization |
| [docs/REPORT.md](docs/REPORT.md)       | the write-up                                      |
| [docs/machine.txt](docs/machine.txt)   | the test machine, and how to dump your own       |

## Characterization

Optimization targets are meaningless without a ceiling to measure against, so the ceiling
is measured first rather than taken from a spec sheet:

- **Compute peak:** 143.88 GFLOP/s, found by sweeping independent FMA chain counts.
The chip needs ~20 in flight to saturate — which is what later motivated v5's wider tile.
- **Memory bandwidth:** ~128 GB/s (STREAM Triad), plateauing from 64 MB/array onward.
- **Ridge point:** 143.88 / 128 ≈ 1.12 FLOP/byte. Square GEMM has intensity N/6, so
  anything past N ≈ 7 is compute-bound. Every size benchmarked here is well past that.

## Verification

Each kernel's output is compared against `gemm_reference` on the same random inputs. The
metric that decides PASS/FAIL is the Frobenius-norm relative error, against a threshold
scaled by `sqrt(K) * u` (the typical fp32 summation rounding bound). Max elementwise
relative error is also reported but not used as the gate: with random-signed data some
reference entries land near zero, and dividing a tiny absolute error by a near-zero value
produces a large number that says nothing about correctness.

Notably, `norm_rel` is unchanged across all six kernels at every size (5.69e-07 at N=1008
for every one of them). The accuracy is set by the arithmetic, not by any of the
structural choices made along the way.

## Status

v5 is current. Next: B is still read from the original matrix on every microkernel call rather than being packed the way A is. Expecting ~3-5 more versions before moving on to general matrix multiplication.