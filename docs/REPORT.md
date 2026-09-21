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
