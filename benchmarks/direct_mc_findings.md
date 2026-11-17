# Engineering status

The prototype implements the CPU R-B reference described in
`direct_mc_protocol.md`. It uses existing native relabeling, circular scanning,
normalization, anchors and alignment, so PWM, BaMM and other supported motif
models share one path. It is serial and keeps query profiles in memory; it is
not yet a production matrix implementation.

Correction: `/dev/kfd` and `/dev/dri` were hidden by the sandbox, not absent
from the host. Outside it, ROCm and OpenCL detect an AMD Radeon RX 9070
(`gfx1201`), with OpenCL reporting 17,095,983,104 bytes of global memory and
a 14,531,585,632-byte maximum allocation. A small OpenCL kernel was compiled
and executed successfully on the GPU. The amount currently free was not
measured. GPU implementation, transfer cost, peak allocation and CPU/GPU
agreement remain to be measured. No acceleration is claimed.

One small CPU smoke timing (one thread, 100 random sequences of 100 bases,
threshold 2, shift range 1, K=3, two example PWM models) measured 0.137 s
for the first pair call with an existing Numba cache, then 0.022 s for 1×1,
0.040 s for 1×2 and 0.045 s for 2×2. These are single measurements on tiny
inputs, not benchmark results or a basis for a 408×408 projection. They
include preparation and all replicates. A cold JIT compilation was not measured.

The 45 focused tests include native versus explicit recoding for PWM, BaMM 0
and 4, and other types, plus prototype collection invariance and identity ties.
They do not establish GPU numerical equivalence or statistical calibration.
The source h11 report contains larger historical CPU timings, but uses its own
implementation and workload and is not a speedup comparison for this code.

## GPU feasibility probe

An experimental OpenCL PWM scanner was run outside the sandbox on RX 9070.
The probe uses 10,000 rows of length 100, width 12, both orientations and
synthetic Float32 weights and bases, including N. It compares identical work
with a single-thread C scanner, with host wall-clock synchronization, and
uploads DNA and weights and downloads all scores on every timed iteration.
The GPU timings below follow one warmup; `setup` includes context creation and
OpenCL program build, potentially using the driver's compilation cache. OpenCL
optimizations were disabled to investigate numerical equivalence. These are
single measurements and the CPU comparator is not the parallel MIMoSA pipeline.

| Models | CPU scan, s | GPU upload + kernel + download, s | GPU setup, s | Explicit device buffers |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.0445 | 0.00285 | 0.157 | 9 MB |
| 16 | 0.672 | 0.0345 | 0.0325 | 129 MB |
| 64 | 2.691 | 0.169 | 0.0318 | 513 MB |

All 162 million GPU scores across these three cases were bitwise identical to
the C reference. The single-model scores were also bitwise identical to the
actual MIMoSA circular scanner on both strands. This covers one synthetic
input, not all floating-point cases. Explicit buffer sizes exclude driver
overhead and are not measurements of peak VRAM. Downloading all 64 model
profiles costs 0.116 s of the 0.169 s, so keeping intermediate data on GPU
may matter. Times do not include CPU-side randomization, logERR, anchors,
alignment, or exceedance counting and must not be called Monte Carlo speedups.

As a separate hybrid correctness check, GPU scores from one synthetic model
were fed into the existing CPU normalization and anchor construction; the
normalized profiles and one J_w comparison with an example PWM were identical
to the CPU path. Three warmed CPU preparations took 0.424–0.433 s, whereas
CPU processing after precomputed GPU scores took 0.376–0.392 s. The latter
does not include GPU work, transfers, orchestration or JIT setup; it is not a
measured full hybrid runtime. In this case scan-only acceleration can remove
only a small fraction of profile preparation, so GPU normalization and
end-to-end measurement are needed before concluding that GPU Monte Carlo is
worth deploying. A separate real-model check showed considerable variation in
CPU preparation, including with eight Numba threads.

The scripts used here are local exploratory files under `experiments/gpu/`,
which is ignored by Git: `scan_probe.c`, `compare_production.py`,
`hybrid_check.py` and `cpu_stages.py`. The GPU executable was built with
`cc -O3 -ffp-contract=off experiments/gpu/scan_probe.c -I/opt/rocm/include
-L/opt/rocm/lib -lOpenCL -lm -o /tmp/mimosa_scan_probe`. Set `MIMOSA_DUMP=1`
for the single-model run before the Python comparison scripts. GPU access
requires execution outside the current sandbox. No end-to-end GPU Monte Carlo,
BaMM kernel, mixed collection, 408×408, CPU/GPU exceedance-count agreement,
peak-memory measurement, or K projection has been measured yet.
