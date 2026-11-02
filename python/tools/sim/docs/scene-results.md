# Scene results for the paper's scenes table

Findings from the production runs of the paper's table scenes, kept for writing the paper. Each section says
which runs it reads, how to rerun them, and how its numbers are computed. The runs themselves are in the
git-ignored `data/profiling/runs/<Scene>/`, next to a `<Scene>.Summary.py` that prints the section's table from
them (`python data/profiling/runs/<Scene>/<Scene>.Summary.py`, from the repo root).

## Celebration (2026-11-01)

**Setup.** `path/to/accelerated-vbd-extras/final_scenes/celebration.h5`: 1e8 Pa letters on a base plate (one mesh,
176,397 nodes, the base fixed) and 36 confetti (537 nodes each) falling on them. Config
`python/tools/sim/configs/scenes/Celebration.toml` at commit fb8516ab:
- 10 ms steps without substeps;
- the VBD family at a fixed 5 x 80 = 400 sweeps per step with `Solver.Convergence.early_exit = 0` (fixed compute, the
  maintainer's choice for this scene);
- Newton, the reference, at a fixed 5 outer iterations x 1 Newton step, CG rtol 1e-6, at most 2000 iterations;
- contact margin 5 mm, VertexSdf band 3 mm, max_dist 5 cm.

5 s per solver on an RTX 5070:

```bash
python -m python.tools.sim.simulate --scene path/to/accelerated-vbd-extras/final_scenes/celebration.h5 \
  --sim-params python/tools/sim/configs/scenes/Celebration.toml --solver <VBD|AAAVBD|Chebyshev|AndersonSolver|Newton> \
  --cd VertexSdf --duration 5 --fps 100 --checkpoint-every 50 --profile-warmup 10 \
  --profile-out data/profiling/runs/Celebration/Celebration.<solver>.Production.json \
  -o data/profiling/runs/Celebration/Celebration.<solver>.Production.h5
python -m python.tools.sim.profiling.trajectory data/profiling/runs/Celebration/Celebration.<solver>.Production.json \
  data/profiling/runs/Celebration/Celebration.<solver>.Production.h5
```

**Results.**

| Solver | Step time (mean) | Letters' largest displacement at 1 / 3 / 5 s | Letter nodes > 5 cm at 5 s | Gradient ratio (median) | Steps with feasibility ratio > 1 (max) | RMS speed at 4-5 s (median) |
|---|---|---|---|---|---|---|
| Newton (5 x 1) | 10.9 s | 14 / 12 / 8 mm | 0 | 27 | 2 (2.63) | 0.005 m/s |
| AAAVBD | 771 ms | 84 / 31 / 34 mm | 0 | 40 | 5 (2.48) | 0.007 m/s |
| Chebyshev | 626 ms | 101 / 279 / 240 mm | 728 | 29 | 3 (1.35) | 0.019 m/s |
| VBD | 627 ms | 820 / 555 / 567 mm | 1,405 | 59 | 27 (3.56) | 0.023 m/s |
| Anderson | 689 ms | 949 / 1055 / 922 mm | 2,274 | 104 | 78 (3.28) | 0.053 m/s |

How the columns are computed:
- **Step time:** the production timing (`step_timeline_ms`) over the 491 steps after the 10 warm-up steps.
- **Letters' displacement:** the largest distance of a letters-and-base node from its rest position, at the frame
  of each time.
- **Gradient and feasibility ratios:** the last outer test of each step, as recorded in the solve status.
- **RMS speed:** `profiling.trajectory`'s mass-weighted RMS speed of the free nodes.

**Findings.**
- The 1e8 Pa letters sag under the VBD family's fixed budget, without any contact involved. AAAVBD stays within
  34 mm of rest after its 160 mm peak at 0.53 s and settles. VBD and Anderson droop by up to 0.8-1 m and are
  still moving at 5 s. Chebyshev starts like AAAVBD (101 mm at 1 s), then droops to about 25 cm. At equal sweeps,
  this is the scene's qualitative difference between the solvers.
- Anderson costs 10% more per step than VBD at the same sweeps: each Anderson iteration costs more than a sweep.
- The gradient ratio doesn't rank the solvers by how much the letters sag: Chebyshev's median (29) is below
  AAAVBD's (40) with 7x its droop, and VBD's (59) is far below what its 0.5 m droop suggests. The residual doesn't
  see this low-frequency error, so the figures should show the shapes or displacements, not only residuals.
- **Newton caveat:** CG reached its 2000-iteration cap in 2400 of the 2455 Newton steps measured, so the reference
  is inexact. Its gradient ratio levels off at about 24 by 3-5 s, against 15 at t = 0. It's still the
  most accurate run by far (letters within 23 mm throughout, peak at 0.42 s).
- Every run is healthy at 5 s: no NaNs, no inverted tets, no confetti through or off the plate (lowest confetti at
  the 5 mm margin). The confetti end up in different places in every run (0.65 m RMS between Newton and AAAVBD):
  the scattering after impact is chaotic, so where they land says nothing about accuracy.
- With early exit off, the solves still test and record their ratios. No VBD-family step met the tolerance at the
  end of its budget, so the table's early-exit column is 0% for this scene.

**Viewing.**

```bash
python -m python.tools.sim.trajectory_viewer data/profiling/runs/Celebration/Celebration.{Newton,AAAVBD,Chebyshev,VBD,AndersonSolver}.Production.h5 \
  --labels Newton AAAVBD Chebyshev VBD Anderson --reference data/profiling/runs/Celebration/Celebration.Newton.Production.h5
```
