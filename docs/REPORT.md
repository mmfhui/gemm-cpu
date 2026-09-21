# GEMM on CPU ~ Report

## Characterization

### **Machine:**
<!-- This project will aim to do this for my M4 pro macbook and my x86 thinkpad. x86 characterization, seperate section, TBD -->
Details used can be found in docs/machine.txt

### **Computed peak (M4 Pro):**
<!-- peak floating-point operations per second (FLOP/s) can be computed by:

clock frequency (clock cycles the processor has each second) multiplied by vector instruction lanes (how many values SIMD can process at once) multiplied by FMA units per cycle (fused multiply-adds for the theoretical max) mulitplied by 2 (as one FMA counts as two operations: one multiply and one add) -->

**FORMULA: peak = clock x lanes x FMA_units_per_cycle x 2**

clock                = ~4.4 GHz   (advertised figure by Apple)

lanes                = 4          (NEON = 128-bit, fp32 = 32-bit, 128/32 = 4)

FMA_units_per_cycle  = UNKNOWN   (not published by Apple, not yet measured)

**therefore,** peak = 4.4e9 x 4 x [?] x 2 GFLOP/s  (unresolved until measured)

### **Arithmetic intensity (Square Matrix Multiplication):**

Square matrix multiplication computes, for every output element: N multiplications (one per term) and N-1 additions to sum them (-1 vanishes as N grows). NxN = N^2 output elements, resulting in 2N^3 FLOPS(N).

A and B are NxN which both need to be read at least once to compute anything: N^2 elements each.
C is NxN which must be written at least once: N^2 elements
In total, the minimum total elements touched is 3N^2.
With each element at fp32 (4 bytes), that is a total of 12N^2 bytes.

Intensity is FLOPs(N) / Bytes(N) and represents the ratio of work done to data touched (low intesisty means memory-bound while high intensity means compute bound).

  **FLOPs(N)** = 2N^3          (one multiplication + one add per (i,j,k) triple)

  **Bytes(N)** = 12N^2         (A, B, C each read/written once, fp32 = 4 bytes)

  **Intensity** = 2N^3 / 12N^2 = N/6 FLOP/byte   -- grows with N

  ### **Measured peak (M4 Pro):**

  **Measured peak:** 143.88 GFLOP/s (CHAINS=20, ITERS=20000000, see compute.cpp comments for more details)

  **Measured bandwidth:** ~128 GB/s (STREAM Triad convention; plateau from 64MB/array onward, see memory.cpp comments)

### **Ridge point**

**Formula: ridge point = peak / bandwidth**
<!-- ridge point is the ratio of math-to-data and helps answer when a workload switches from being limited by one ceiling to the other (below is less math per byte so the data part is slower and waiting on bandwith: memory-bound, above is more math per byte so the math part is slower and waiting on peak: compute-bound) -->

ridge point = 143.88 / 128 ≈ 1.12 FLOP/byte

<!-- compare to intensity as intensity is the same ratio except for the specific workload -->

**Comparing** to intensity = N/6:

  N/6 = 1.12  ->  **N ≈ 6.75**

**Verdict: compute-bound for N >= ~7, ceiling = 143.88 GFLOP/s.**

## Implementation Log

### v0 ~ naive triple loop (ijk)

<!-- Predicted: low single digits of peak, because B is accessed with stride
ldb in the innermost loop (almost every read lands on a fresh cache
line). -->

| N    | GFLOP/s | % peak | max_rel  | norm_rel | threshold | status |
|------|---------|--------|----------|----------|-----------|--------|
| 16   | 4.571   | 3.18%  | 3.302e-05| 7.586e-08| 4.768e-06 | PASS   |
| 64   | 4.636   | 3.22%  | 1.225e-04| 1.487e-07| 9.537e-06 | PASS   |
| 112  | 4.172   | 2.90%  | 6.573e-04| 1.914e-07| 1.262e-05 | PASS   |
| 240  | 3.820   | 2.65%  | 4.150e-02| 2.791e-07| 1.847e-05 | PASS   |
| 496  | 3.136   | 2.18%  | 7.977e-02| 3.974e-07| 2.655e-05 | PASS   |
| 1008 | 2.803   | 1.95%  | 4.533e-01| 5.694e-07| 3.785e-05 | PASS   |

Result: All sizes PASS. % of peak falls as N grows (3.18% -> 1.95%), consistent with the diagnosed cause: as N grows, more
of B's working set exceeds cache, so a shrinking fraction of each fetched cache line is used before eviction.

<!-- max_rel grows to 45% at N=1008 while norm_rel stays ~5.7e-7 (expected):
with >1M random-signed output elements, some Cref entries land near
zero by chance, and dividing a tiny absolute error by a near-zero true
value produces a large but meaningless relative number on that one
element. norm_rel is the metric that reflects actual correctness here. -->

Trigger for v1: naive access pattern is the diagnosed bottleneck, requiring reordering to fix it.

### v1 ~ loop reorder (ikj)

<!-- Predicted: __ % of peak: unit-stride B and C should give a
large jump over v0, but still far from ceiling since nothing here holds
values in registers across iterations or uses more than one SIMD lane. -->

| N    | GFLOP/s | % peak | max_rel   | norm_rel  | threshold | status |
|------|---------|--------|-----------|-----------|-----------|--------|
| 16   | 12.300  | 8.55%  | 2.863e-05 | 7.916e-08 | 4.768e-06 | PASS   |
| 64   | 30.691  | 21.33% | 4.570e-04 | 1.441e-07 | 9.537e-06 | PASS   |
| 112  | 39.575  | 27.51% | 5.380e-04 | 1.928e-07 | 1.262e-05 | PASS   |
| 240  | 35.112  | 24.40% | 9.649e-02 | 2.793e-07 | 1.847e-05 | PASS   |
| 496  | 34.820  | 24.20% | 6.589e-01 | 3.980e-07 | 2.655e-05 | PASS   |
| 1008 | 34.010  | 23.64% | 1.548e+00 | 5.688e-07 | 3.785e-05 | PASS   |

Result: Roughly 10x over v0 at every size except N=16, where both versions already fit comfortably in cache and the stride fix has less to win.
norm_rel essentially unchanged from v0 (e.g. 5.688e-7 vs 5.694e-7 at N=1008), confirming the reordered summation doesn't degrade accuracy.

<!-- Note: v0's numbers shifted slightly between this run and the previous
session (e.g. N=112: 4.172 -> 3.291 GFLOP/s), a ~20% swing on an
unchanged kernel -- session-level noise, doesn't affect the v0-vs-v1
conclusion but worth flagging. Re-check power state before the next run. -->

Trigger for v2: unit-stride access is fixed, but nothing yet keeps values in registers across the k-loop or uses SIMD explicitly.
