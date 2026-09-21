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

### v2 ~ register-tiled 4x4 microkernel, NEON intrinsics

<!-- Predicted: [your number] -- C's memory traffic drops from K touches to
1 per tile, and computation moves from 1 number at a time to 4. -->

| N    | GFLOP/s | % peak | norm_rel  | status |
|------|---------|--------|-----------|--------|
| 16   | 49.349  | 34.30% | 8.616e-08 | PASS   |
| 64   | 59.356  | 41.25% | 1.481e-07 | PASS   |
| 112  | 54.871  | 38.14% | 1.894e-07 | PASS   |
| 240  | 43.403  | 30.17% | 2.789e-07 | PASS   |
| 496  | 39.032  | 27.13% | 3.982e-07 | PASS   |
| 1008 | 36.824  | 25.59% | 5.692e-07 | PASS   |

Result: Large jump at small N (~doubled v1 at N=64), but the gain erodes as N grows. At N=1008, v2 is barely ahead of v1 (25.6% vs 24.2%).
norm_rel matches v0 and v1 at every N, confirming error is governed by the arithmetic (sqrt(K)*u), not by loop structure.

Trigger for v3: the erosion at large N means the data no longer fits in cache. Currently, nothing blocks the loops so that A/B/C tiles stay resident while they're being reused.

### v3 ~ cache blocking (BLOCK=96)
<!-- **Tile-size prediction**

**Step 1: bytes per tile**

A tile is b x b elements. At fp32 (4 bytes/element):

  bytes_per_tile = b^2 x 4

**Step 2: bytes for three tiles together**

Blocking needs A's tile, B's tile, and C's tile resident simultaneously:

  bytes_total = 3 x 4 x b^2 = 12 b^2

**Step 3: the capacity constraint**

This total can't exceed the actual cache size. We will call it M_cache (bytes).

  12 b^2 <= M_cache

  b^2    <= M_cache / 12

  b      <= sqrt(M_cache / 12)

**Step 4: plug in measured numbers**

L1d = 131072 bytes, measured this session via sysctl (docs/machine.txt):

  b <= sqrt(131072 / 12)

  b <= sqrt(10922.67)
  
  b <= 104.5

**Step 5: Round to usable number**

104.5 isn't a usable block size on its own: it needs to be a whole number, and ideally a multiple of 4, since the microkernel operates in 4x4 chunks. A block
size not divisible by 4 would create a partial microkernel call at every single block boundary, adding edge-case complexity for no benefit at this stage.

104 itself is already divisible by 4 (104/4 = 26), so it's directly usable. But the theoretical bound of 104.5 assumes the ENTIRE L1d is
available for exactly these three tiles and nothing else, which isn't true in practice (Real caches are set-associative, not fully associative and other temporaries also occupy cache lines):

For these reasons, the practical starting choice is below the theoretical ceiling, not at it. Starting value: BLOCK = 96 (also a
multiple of 4), leaving headroom for the effects above.

**Scope:**

**v3 BLOCK sweep predicted vs measured**

Predicted: BLOCK ~ 96 (derived: b <= sqrt(131072/12) ~ 104, rounded down
for associativity/occupancy headroom).

Measured (swept 32/48/64/80/96/112/128, N=1008):
  32:  44.44%   <- best
  48:  37.46%
  64:  35.62%
  80:  34.28%
  96:  34.48%   (predicted value)
  112: 33.99%
  128: 32.14%

Result: prediction was wrong, and wrong in the opposite direction from
its own caveat -- the derivation assumed smaller-than-theoretical would
be needed for safety margin, but the real optimum (32) is roughly 3x
smaller than even the conservative starting guess, not just modestly
below the theoretical ceiling (104.5).

Reconciliation: the b <= sqrt(M/12) bound only enforces that three
tiles fit ONCE. It doesn't account for how long they need to stay
resident -- a block of size b requires (b/4)^2 microkernel calls per
K-block, all needing the same A/B panels to survive in cache
simultaneously. Larger blocks hold more data resident for
proportionally longer, giving far more opportunity for eviction via
set-associativity conflicts and contention from other cores sharing
the same L2 cluster (see Phase 1: L2 is shared across all 8 P-cores,
not private). The capacity model was necessary but not sufficient --
it ignored residency duration entirely.

Decision: BLOCK = 32, chosen empirically, overriding the derived value.

This derivation blocks for L1 only. L2 (16 MB, shared across the whole P-cluster) is not yet layered on top as v3 is L1 blocking specifically. -->

<!-- Predicted: erosion seen in v2 at large N should be reduced or
eliminated, since A/B/C tiles now stay within L1-sized blocks. -->

| N    | GFLOP/s | % peak | norm_rel  | status |
|------|---------|--------|-----------|--------|
| 16   | 39.385  | 27.37% | 7.815e-08 | PASS   |
| 64   | 68.759  | 47.79% | 1.476e-07 | PASS   |
| 112  | 71.740  | 49.86% | 1.947e-07 | PASS   |
| 240  | 70.291  | 48.85% | 2.807e-07 | PASS   |
| 496  | 66.345  | 46.11% | 4.001e-07 | PASS   |
| 1008 | 64.779  | 45.02% | 5.693e-07 | PASS   |

Result:  erosion from v2 is essentially eliminated. v3 declines only 49.86% -> 45.02% from N=112 to N=1008, versus v2's 38.17% -> 25.42% over the same range. v3 is now ahead of v2 at every size except N=16.

Trade-off: at N=16 only v3 underperforms v2 here (27.37% vs 45.55%), because BLOCK=32 does not evenly divide 16, so the blocking loops produce a single undersized, fragmented block rather than cleanly containing the whole matrix (pure overhead with no benefit at sizes that didn't need protecting from eviction).

Trigger for v4: A is still re-fetched with strided scalar loads inside the microkernel rather than being copied into a contiguous buffer once
per block (what packing fixes).

### v4 -- packing (BLOCK=32)
<!-- Predicted: [your number] -- expected real gains at larger N where
blocks are reused enough to amortize the copy cost; possible wash or
loss at N=16 given v3's fragmentation problem at that size. -->

| N    | GFLOP/s | % peak | norm_rel  | status |
|------|---------|--------|-----------|--------|
| 16   | 49.054  | 34.09% | 7.596e-08 | PASS   |
| 64   | 71.899  | 49.97% | 1.470e-07 | PASS   |
| 112  | 74.516  | 51.79% | 1.916e-07 | PASS   |
| 240  | 73.353  | 50.98% | 2.780e-07 | PASS   |
| 496  | 71.866  | 49.95% | 3.988e-07 | PASS   |
| 1008 | 65.124  | 45.26% | 5.694e-07 | PASS   |

Result: v4 beats v3 at every size, including N=16 (34.09% vs 27.37%). Packing's benefit (sequential A reads instead of strided) apparently
doesn't require many block-reuses to pay off, it even helps a single fragmented block. Packing's win is not just primarily about reuse amortization here but also about removing strided access from the hot loop directly. Peak achieved: 51.79% at N=112. The remaining gap to 100% would require multiple microkernel shapes, K/M/N-specific block tuning, or multi-threading.

<!-- norm_rel unchanged from v3 at every size (packing doesn't alter the
arithmetic or its order) -- confirms the gain is purely structural.
max_rel roughly doubled at some sizes (e.g. N=1008: 3.19e-01 ->
6.17e-01); given norm_rel is flat, this is the same near-zero-reference
phenomenon from prior rungs landing on a different element, not a new
correctness issue. -->

Trigger for v5: the 4x4 tile gives only 4 independent accumulator chains  which is short of the ~20 compute.cpp's own peak sweep found
necessary to saturate the FPU. Widening the microkernel would be the next optimization.

### v5 ~ widened microkernel, 8x8 tile (16 accumulators)
<!-- Predicted: [your number] -- targeting closer to the ~20 independent
FMA chains compute.cpp found necessary for this chip's peak throughput,
up from v4's 4. -->

| N    | GFLOP/s | % peak | norm_rel  | status |
|------|---------|--------|-----------|--------|
| 16   | 49.054  | 34.09% | 7.453e-08 | PASS   |
| 64   | 114.398 | 79.51% | 1.500e-07 | PASS   |
| 112  | 115.670 | 80.39% | 1.947e-07 | PASS   |
| 240  | 113.622 | 78.97% | 2.813e-07 | PASS   |
| 496  | 112.722 | 78.34% | 3.985e-07 | PASS   |
| 1008 | 104.268 | 72.47% | 5.692e-07 | PASS   |

Result: ~1.5-1.6x over v4 at every size N>=64, reaching 80.39% at N=112. v4's bottleneck seems to be FPU occupancy, not memory traffic, and closing most of the gap to compute.cpp's ~20-chain finding closed most of the gap to peak. An exception was that N=16 barely moved (27.37% -> 34.09%), now the worst
performer in the table (an 8x8 tile with K too shallow at this size to amortize the wider tile's fixed overhead against). Consistent with
the pattern seen at prior iterations: larger structural optimizations trade a fixed cost against N=16's very small amount of real work. 

Trigger for v6: B is still read directly from the original matrix on every microkernel call rather than packed into a contiguous buffer the
way A is (microkernel's memory access pattern).

<!-- norm_rel essentially unchanged across all six kernel versions at every
size tested (e.g. N=1008: 5.694e-07 across v0-v5) -- six independent
implementations, one accuracy profile, governed entirely by the
arithmetic (sqrt(K)*u), not by any structural choice made along the way. -->