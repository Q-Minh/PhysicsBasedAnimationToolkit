# GPU Anderson acceleration: implementation notes

Pitfalls we hit while porting Anderson-accelerated VBD (C++ `source/pbat/sim/algorithm/vbd/Anderson.h`) to the GPU
(`python/tools/sim/gpu/vbd/aasolver.py`), with the evidence behind each fix. These notes are kept for the paper's
discussion of the Anderson baseline: each entry says what went wrong, why, what we changed, and what remains.

Notation: window size `m`, active window `mk = min(m, k)`, residual `f_k = x_k+1 - x_k` (one fixed-point
application), `F` the `(3N, mk)` matrix of residual differences, `gamma` the least-squares mixing coefficients,
`p = vbd_iters_per_anderson_iter` the VBD sweeps per Anderson iteration.

## 1. The least-squares solve needs a relative, adaptive regularization

**What the CPU does.** C++ solves `min |F gamma - f|` with Eigen's complete orthogonal decomposition (COD) of `F`,
whose threshold `codNumericalZero` (1e-10) is relative to the largest pivot.

**First GPU attempt.** The GPU solves the normal equations `F^T F gamma = F^T f` instead (CuPy's cuSOLVER, which
would provide a QR or COD, refuses to run during stream capture). It added
`cod_numerical_zero` to `F^T F` as an *absolute* ridge. On real scenes the Gram diagonal is ~1e-13, so the ridge
dominated in more than 99% of mixes: `gamma` collapsed to ~0, and Anderson silently reduced to plain VBD.

**Fix** (commit `0671d4b2`). Scale the system by its largest diagonal entry, factorize it **unregularized** with a
Cholesky, and refactorize with a ridge only if a pivot fails: `L_jj^2 <= max(tau^2, m eps_f32)`, `tau =
cod_numerical_zero`. The ridge is then relative to `max_j (F^T F)_jj`, like the COD's relative singular-value
cutoff, and the `m eps_f32` floor accounts for float32 rounding of the Gram entries (squaring `F` into the normal
equations squares its condition number, so relative pivots below that level are noise). Zero histories and non-finite
solutions fall back to the plain VBD step (`gamma = 0`). A Jacobi-preconditioned CG on the same system is available as
an alternative (`AndersonLinearSolver`).

**Evidence.** MediumLongBendingBeam: the ridge now fires in 0.2% of mixes, and Anderson reaches ~3.9e-6 median
per-sweep residual vs 1.3e-5 for VBD, at 3-5% overhead per iteration. In the 2026-10-20 trampoline diagnosis (item 3)
the ridge never fired and `cond(F^T F / max diag) <= ~20`, so the regularization is not what limits accuracy there.

**For the paper.** The regularization must be scale-invariant: residual magnitudes vary by orders of magnitude across
scenes, time steps and iterations, so any absolute threshold is wrong somewhere.

## 2. The Gram reduction must be deterministic

**Problem.** The Gram matrix `[F f]^T [F f]` is a reduction over `3N` entries. The GPU summed per-tile products with
atomic adds (`wp.tile_atomic_add`), whose order changes from run to run, so `G` differed in its last bits between
identical runs.

**Why it matters.** Anderson amplifies those bits. Near the float32 floor (item 3) the least-squares fit is
ill-determined, and last-bit differences in `G` change `gamma` enough to send identical solves down different paths.
On trampolines-test Y1e7, h = 1e-2, p = 1, 40 repeated solves of the set's two problems took a median 13.4 ms to reach 99%
error reduction, but 11 of 40 solves were more than 2x slower, the worst 194 ms (13-15x). The same solve's final
error after 1000 iterations ranged from 3.3e-3 to 8.5e-3 across runs. No run was reproducible, so per-solve timings
were not either, and captured-graph and eager runs could only be compared "within Anderson's noise".

**Fix** (commit `985cd753`). Two passes: each tile writes its `(m+1)^2` product to its own row of a buffer, and a
second kernel sums the rows in a fixed order (one 32-thread block per entry of `G`). Anderson is now bit-reproducible
within a process, across processes, and between captured and eager execution.

**Cost.** ~2 µs per Anderson iteration on the trampoline (10,643 nodes): +2.1-2.2% at p = 1, +1.5-1.9% at p = 2. A
single-block reduction cost ~3 µs; the cost is mostly the extra kernel launch.

**What it does not fix.** Determinism removes the run-to-run variance, not the stall itself: a problem that lands in
the slow mode now does so every time. Accumulating the Gram in float64 didn't help either (item 3).

**For the paper.** The noise didn't trap the solver in a local minimum of the energy. It perturbed the mixing
coefficients, which decided whether the fast phase ended at ~0.5% or ~1.8% error. Any GPU Anderson implementation
that uses atomics in its reductions inherits this nondeterminism, and benchmarks of it measure a distribution, not a
number.

## 3. Anderson stalls at a float32 precision floor

**Observation.** On stiff dynamic problems (trampoline Y1e7, h = 1e-2) Anderson converges fast, then stalls between
~0.5% and ~1.8% relative error.

**Mechanism** (2026-10-20 diagnosis). At the stall, one sweep's displacement `|f|` is ~5e-5 m over positions of
~1.4 m rms: about 2 float32 ulps per entry. `f = x - x_prev` is computed in float32, so rounding noise is ~35% of
`|f|`, and the least-squares problem fits noise. At the stall `sum(gamma)` jumps from about -1 to about +2. The
regularization never fires and the system is well conditioned (item 1), and a float64 Gram doesn't help: the
noise is already in `F` and `f`, not in the reduction. The softer Y1e6 problems showed no slow solves (0 of 40).

**Mitigation.** Larger residuals per Anderson iteration push the floor down: `p` VBD sweeps per Anderson iteration
(`vbd_iters_per_anderson_iter`, commit `0a3cca51`) make `f` p sweeps long. On the trampoline pilot (5 scenes, Newton's
time budgets) p = 2 bounds the tail (no solve 4x slower than typical in 60+ solves; p = 1's worst was 13-15x) and gets
further on the hard sets (Y1e8 dynamic 90% vs 70% error reduction, heterogeneous 95% vs 85%, lower final errors on
every static scene), at +7-25% time to 95% and +14-78% to 99% on the easy dynamic sets. The ~1% floor remains with
p = 2: Y1e7 lands in a ~1.8x slower mode in 6 of 10 solves. The benchmark uses p = 2
(`configs/benchmark/shared.toml`); this is close to periodic Anderson acceleration, which applies several fixed-point
steps per Anderson step. Adapting p, or restarting the history near the floor, would be a new Anderson-VBD variant and
is out of scope.

**For the paper.** Anderson's 99% numbers on stiff scenes at h = 1e-2 sit at a float32 floor, which is a limitation of
Anderson in single precision rather than of the benchmark. It's a weakness of the baseline to report, not to tune
away.

## 4. GPU engineering constraints

These shaped the implementation but don't change the algorithm:

- **CUDA graph capture.** The whole solve is one captured graph. The window fills up over the first `m` iterations,
  so `mk` is a runtime argument and inactive slots are masked to solve to exactly 0. One graph serves every iteration,
  and the first `m` iterations are recorded separately from the replayed block.
- **No cuBLAS/cuSOLVER through CuPy.** CuPy raises `NotImplementedError` for them during stream capture. The Gram,
  Cholesky and solve use Warp tile operations (MathDx), which capture cleanly.
- **Kernels specialized per `m`.** Tile shapes are compile-time constants, so the kernels are generated per window
  size and built before capture starts.
- **Equal-work comparisons.** With p > 1, an Anderson iteration is not a sweep. Budgets count VBD sweeps for every
  VBD-family solver (`n // p` Anderson iterations, then `n % p` plain sweeps; commit `c3cca7e5`).
