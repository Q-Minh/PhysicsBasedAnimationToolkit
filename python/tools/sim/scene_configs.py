"""TEMPORARY: write the per-scene configs ``configs/scenes/*.toml`` from the paper configs and the maintainer's rules.

Like ``paper_configs.py``, this module exists only until the paper's parameter files are finalized; then remove it.

Each scene config is a complete, explicit parameter set (every serialized key, like a paper config), so it can be
diffed against ``configs/paper`` when the paper configs are overwritten. It is its source paper config plus the
overrides in :data:`SCENES`, each with its reason; the file's header lists every key that differs from the source,
grouped by reason, with the source's value.

Usage, from the repository root (no GPU needed):

    python -m python.tools.sim.scene_configs make
    python -m python.tools.sim.scene_configs check

``make`` writes ``configs/scenes/<scene>.toml``. ``check`` verifies that each scene config holds exactly the
parameters ``make`` would write, and every key of ``configs/defaults.toml``; it exits nonzero otherwise. Rerun
``make`` after ``paper_configs make`` (a paper config or a serialized parameter changed).
"""

import argparse
import datetime
import pathlib
import sys
import textwrap
import tomllib
from dataclasses import dataclass, field

import numpy as np

from . import config

CONFIGS = pathlib.Path(__file__).parent / "configs"
VBD_FAMILY = ("AAAVBD", "VBD", "Chebyshev", "AndersonSolver")


@dataclass
class Rule:
    """Keys set together for one reason: ``{"Table.Sub.key": value}``."""

    reason: str
    values: dict


@dataclass
class Scene:
    source: str  # configs/paper/<source>.toml, or "defaults" (configs/defaults.toml) for a scene without a paper config
    scene_files: str  # which scenes it is for
    rules: list[Rule]
    notes: list[str] = field(default_factory=list)


def _per_solver(key: str, value, solvers=VBD_FAMILY) -> dict:
    return {f"Solver.{s}.{key}": value for s in solvers}


# The maintainer's durable rules (plans/MASTER.md, sim-run skill "Tuning")
PENALTY = Rule(
    "Adaptive contact penalty per time step (TIMESTEP, the code default; maintainer: CONSTANT only with a stated "
    "justification), with the code's gamman_adaptive / gammaf_adaptive, instead of the paper's constant gamman.",
    {"Contact.ContactDynamics.penalty_adaptivity": 1},
)
DMIN = Rule(
    "Positive AL contact margin dmin = 1 mm (the paper's -1e-4 dates from penetration-only VertexSdf detection): "
    "resting vertices stay outside their partner, where the SDF sign is reliable, detection (below dmin + band) "
    "starts before contact, and the feasibility tolerance (ftol * dmin) is meaningful.",
    {"Contact.ContactDynamics.dmin": 1e-3},
)
BAND = Rule(
    "VertexSdf band 2 mm, close to dmin (maintainer): a 2 cm band made the sphere stick to the trampolines under "
    "Newton, and the 1 mm default once let bodies pass through 10 cm plates. max_dist is kept.",
    {"Contact.CDType.VertexSdf.band": 2e-3},
)
NEWTON_CONTACT = Rule(
    "Newton with contact: 5 outer iterations (contact detection passes) x 2 Newton steps (maintainer).",
    {"Solver.Newton.n_max_iters": 5, "Solver.Newton.newton.n_max_iters": 2},
)
INVERSE = Rule(
    "Inverse vertex solve for the VBD family (maintainer: EVD only when singular vertex Hessians cause problems).",
    _per_solver("vlinsolve", 0),
)
GAMMAS = Rule(
    "AAAVBD's shared parameter set, the same in every scene (gammadown/gammadownS 0.99/0.95, gammaup/gammaupS "
    "0.5/0.3): the Lipschitz normalization is meant to make them problem-invariant.",
    {
        "Solver.AAAVBD.gammadown": 0.99,
        "Solver.AAAVBD.gammadownS": 0.95,
        "Solver.AAAVBD.gammaup": 0.5,
        "Solver.AAAVBD.gammaupS": 0.3,
    },
)
CONTACT = [PENALTY, DMIN, BAND, NEWTON_CONTACT]
NEWTON_HET_CONTACT = Rule(
    "Newton on heterogeneous soft-stiff contact: 30 outer iterations (contact detection passes) x 1 Newton step, "
    "with strict CG (rtol 1e-9, at most 4000 iterations) (maintainer, 2026-10-20): with 5 x 2 and the default CG, "
    "the sphere's contact with the stiff trampoline center was unstable; 30 x 1 with strict CG is stable, if slow.",
    {
        "Solver.Newton.n_max_iters": 30,
        "Solver.Newton.newton.n_max_iters": 1,
        "Solver.Newton.LinearSolver.rtol": 1e-9,
        "Solver.Newton.LinearSolver.n_max_iters": 4000,
    },
)
HET_CONTACT = [*CONTACT, NEWTON_HET_CONTACT]
SOLVERS = [INVERSE, GAMMAS]


def _budget(sweeps: int, why: str, paper: str) -> Rule:
    """No substeps and 5 outer iterations (contact detection passes) x ``sweeps`` for every VBD-family solver."""
    return Rule(
        f"{why}: no substeps, 5 x {sweeps} = {5 * sweeps} sweeps per step for every VBD-family solver (the paper "
        f"file gives {paper}).",
        {"Integration.substeps": 1, **_per_solver("n_max_iters", 5), **_per_solver("n_subproblem_max_iters", sweeps)},
    )


def _stiff_budget(paper: str) -> Rule:
    return _budget(80, "The maintainer's budget for stiff heterogeneous scenes", paper)


TRAMPOLINE_DETECTION = Rule(
    "max_dist 50 cm, scaled to the metre-scale trampoline geometry (10 cm plate) and the sphere's impacts "
    "(maintainer, 2026-10-21; was 30 cm).",
    {"Contact.CDType.VertexSdf.max_dist": 0.5},
)
SPHERE_BAND = Rule(
    "VertexSdf band 1 cm for the sphere falling on the trampolines (maintainer, 2026-10-21): at 2 mm the sphere's "
    "impact on the heterogeneous circle trampoline exploded, and at 5 mm VBD and AAAVBD still exploded on the "
    "second bounce without substeps.",
    {"Contact.CDType.VertexSdf.band": 1e-2},
)
TRAMPOLINE = [TRAMPOLINE_DETECTION, SPHERE_BAND]
ARMADILLO_SPAGHETTI_CONTACT = Rule(
    "Contact for the armadillo spaghetti scene: VertexSdf band 5 mm and a 1 cm AL margin dmin (maintainer, "
    "2026-10-21), with the source's 5 cm VertexSdf max_dist. The scene file's floor and ceiling planes "
    "sit 2 cm from the pillars' ends, beyond the detection distance dmin + band = 1.5 cm, so they report no contacts. "
    "max_dist was 20 cm until 2026-10-26 (maintainer, 2026-10-26): 5 cm still covers any vertex inside a pillar (10 "
    "cm wide), is over 3x the deepest penetration in any DragonImpact run (1.49 cm), and saves about 21% of "
    "DragonImpact's step at 7.0 s with identical pairs.",
    {"Contact.CDType.VertexSdf.max_dist": 0.05, "Contact.CDType.VertexSdf.band": 5e-3, "Contact.ContactDynamics.dmin": 1e-2},
)
DRAGON_IMPACT_BUDGET = Rule(
    "10 substeps per 10 ms step for the dragons' impact on the armadillo spaghetti (maintainer, 2026-10-23), each of "
    "1 outer iteration x 100 sweeps for every VBD-family solver: at 5 substeps of 1 x 80, the "
    "pillars jittered after the impact (mid-span RMS speed 0.8-2.2 m/s, velocity reversal 0.7-1.0, rejected by the "
    "maintainer, against 0.35-0.55 m/s and 0.4-0.5 at 10 substeps, AAAVBD with the twist stopped at 5.0 s), and a "
    "thin feature at the left dragon's trailing end (375 nodes) settled into a flapping limit cycle (about 8 Hz, 20 cm "
    "range, 4-5 m/s, with the twist stopped at 4.0 or 5.0 s) that decays at 10 substeps (2.4 to 0.14 m/s over 7 s, "
    "no 8 Hz component). Newton gets 20 outer iterations per step, 2 per substep with 2 Newton steps and "
    "CG rtol 1e-8, at most 4000 iterations; untested on this scene.",
    {"Integration.substeps": 10, **_per_solver("n_max_iters", 1), **_per_solver("n_subproblem_max_iters", 100),
     "Solver.Newton.n_max_iters": 2, "Solver.Newton.newton.n_max_iters": 2,
     "Solver.Newton.LinearSolver.rtol": 1e-8, "Solver.Newton.LinearSolver.n_max_iters": 4000},
)
DRAGON_IMPACT_FAST_BUDGET = Rule(
    "A lower budget for the dragons' impact, next to ArmadilloSpaghetti.DragonImpact's 100 sweeps (maintainer, "
    "2026-10-27): 40 sweeps per substep for every VBD-family solver, with 0.9 of the AL multiplier step. From t = 0 "
    "at 40 sweeps, AAAVBD reached 10 s with multiplier_step 0.75 and 0.9 and diverged at the dragons' impact (steps "
    "479-480) with 0.25, 0.5 and 1. At 0.9 it then ran on to 18 s with no fast nodes, penetration streak or dropped "
    "pairs after 10 s: its median gradient ratio over 10-18 s was 1037 (688 at 100 sweeps), its worst feasibility "
    "ratio 0.38 (0.37), at most 9 inverted armadillo elements (5), and its pillars and armadillo moved about half as "
    "fast at 17-18 s, on a trajectory that differs from the 100-sweep run's since the twist. "
    "multiplier_step is a contact parameter, shared by every solver (docs/solver-accuracy-notes.md, entries 12, 14 "
    "and 15).",
    {**_per_solver("n_subproblem_max_iters", 40), "Contact.ContactDynamics.multiplier_step": 0.9},
)
ARMADILLO_GAMMAS = Rule(
    "AAAVBD's acceleration more conservative than the shared set for the armadillo (maintainer, 2026-10-21; a "
    "documented per-scene exception): pulled hard, then suddenly released, its energy changes abruptly, where AAAVBD "
    "needs gammadown/gammadownS 0.75 and gammaup/gammaupS 0.5 to stay stable, the most aggressive values that do. It "
    "shows AAAVBD's limitation and its controllability.",
    {"Solver.AAAVBD.gammadown": 0.75, "Solver.AAAVBD.gammadownS": 0.75, "Solver.AAAVBD.gammaup": 0.5,
     "Solver.AAAVBD.gammaupS": 0.5},
)
ARMADILLO_BUDGET = Rule(
    "The maintainer's budget for the armadillo (2026-10-21; no contact): 1 outer iteration x 1000 sweeps for every "
    "VBD-family solver, and Newton 1 x 30 Newton steps (usually ending early; after the release it sometimes runs all "
    "30 with a small residual, line search keeping it stable; CG mostly ends under 1000 iterations, so the CG "
    "defaults are kept).",
    {**_per_solver("n_max_iters", 1), **_per_solver("n_subproblem_max_iters", 1000), "Solver.Newton.n_max_iters": 1,
     "Solver.Newton.newton.n_max_iters": 30},
)
CAPSULE_CONTACT = Rule(
    "Contact scaled to the capsule bowl's size, about 8.8 m across (maintainer, 2026-10-21): VertexSdf max_dist 50 cm "
    "and band 1 cm, and a 1 cm AL margin dmin; the friction (mu_f 0.2) it was tested with, the code default.",
    {"Contact.CDType.VertexSdf.max_dist": 0.5, "Contact.CDType.VertexSdf.band": 1e-2, "Contact.ContactDynamics.dmin": 1e-2,
     "Contact.ContactDynamics.mu_f": 0.2},
)
CAPSULE_BUDGET = Rule(
    "The maintainer's budget for the capsule bowl (2026-10-21): no substeps, 10 outer iterations (contact detection "
    "passes) x 20 sweeps = 200 sweeps per step for every VBD-family solver, and Newton 10 x 2 (the paper file gives 5 "
    "substeps of 3 x 3 sweeps).",
    {"Integration.substeps": 1, **_per_solver("n_max_iters", 10), **_per_solver("n_subproblem_max_iters", 20),
     "Solver.Newton.n_max_iters": 10, "Solver.Newton.newton.n_max_iters": 2},
)
BASKET_BUDGET = Rule(
    "One budget for every VBD-family solver on the basket of fruit, so that the paper shows their different "
    "behaviour at equal budget (maintainer, 2026-11-03): 5 substeps per 10 ms step, each of 3 outer iterations "
    "(contact detection passes) x 50 sweeps, 750 sweeps per step (the paper file gives 3 substeps of 10 x 15 = 450). "
    "With 10 x 15 per substep, AAAVBD kept the fruit in the basket to 2 s at 3 substeps, while VBD blew up at 3 and "
    "4 substeps and ran to 2 s at 5. Fewer, longer subproblems also suit AAAVBD, whose first sweep of every "
    "subproblem is plain VBD. At 750 sweeps per step (2026-11-03), AAAVBD ran to 2 s with every fruit in the basket "
    "at 1 x 150, 2 x 75 and 3 x 50 (896, 912 and 934 s, against 1103 s at 10 x 15), but VBD blew up at 1 x 150 and "
    "2 x 75 (0.73 to 0.79 s) and ran to 2 s at 3 x 50 (853 s, against 1041 s at 10 x 15), with 7 of the 9 fruit "
    "falling through its strands. Each outer iteration per substep costs about 25 ms per step. Lower budgets fail "
    "with VBD: it blew up at 1 x 80, 2 x 40, 3 x 27 (400 to 405 sweeps per step, 0.82 to 0.86 s) and 10 x 9 (450, "
    "0.78 s), while AAAVBD ran to 2 s at 1 x 80 and 10 x 9 with every fruit in the basket. Newton keeps its "
    "settings, now per substep (untested at 5 substeps).",
    {"Integration.substeps": 5, **_per_solver("n_max_iters", 3), **_per_solver("n_subproblem_max_iters", 50)},
)

FOUR_BUNNIES_BDF2 = Rule(
    "BDF2 time integration for the four bunnies' bounce (maintainer, 2026-11-09): BDF1 at h = 10 ms damps the "
    "rebound numerically. Rebound height over drop height of the bunnies' centers of mass (AAAVBD, 5 x 20 sweeps, "
    "the source file's other settings with this scene's rules): on the source scene (every body 1000 kg/m^3) 0.33 "
    "with BDF1, 0.48 with BDF2, 0.50 with 5 substeps of BDF1 (h = 2 ms) and 0.34 with 5x the sweeps (every step "
    "converged early, so it isn't solver error); with the 250 kg/m^3 trampoline 0.41 with BDF1 and 0.65 with BDF2 "
    "(first bounce 51 cm clear of the trampoline, a second of 45 cm). Data: "
    "data/profiling/investigations/FourBunnies-density-tuning.",
    {"Integration.bdf_scheme": 2},
)

BASKET_MARGIN = Rule(
    "AL contact margin dmin = 2 mm for the woven basket (maintainer, 2026-11-05; band unchanged at 2 mm): its "
    "strands are at least 8.7 mm in half-thickness, the gaps between strands are 4.7 mm at the 0.1 percentile "
    "(min 0.40 mm), and only about 35 of the 346k strand vertices start within 2 mm of another strand, so the "
    "detection reach dmin + band = 4 mm stays under the 0.1-percentile gap. BasketOfOranges' and EmptyBasket's "
    "probable sources (extras final_scenes) have the same 16 strand meshes as basket_of_fruit (same vertices).",
    {"Contact.ContactDynamics.dmin": 2e-3},
)


# Contact pair capacities, as multipliers of the contact multimesh's vertices (its surface vertices)
CONTACT_CAPACITIES = Rule(
    "Contact pair capacities sized for every surface vertex in contact at once, each with one partner (VertexSdf "
    "gives a vertex at most one pair per other component): VF pairs for all of them, VE for half and VV for a sixth, "
    "a closed triangle mesh's feature shares (V : E : F = 1 : 3 : 2, by Euler's formula with E = 3V and F = 2V; the "
    "measured peaks were at most 39% VE and 7% VV pairs), and no EE pairs, which VertexSdf doesn't detect (the "
    "storage keeps 1 slot) (maintainer, 2026-10-22: conservatively rather than minimally; dropped pairs now warn).",
    {
        "Contact.ContactStorage.n_vf_contact_capacity": 1.0,
        "Contact.ContactStorage.n_ve_contact_capacity": 0.5,
        "Contact.ContactStorage.n_vv_contact_capacity": 0.17,
        "Contact.ContactStorage.n_ee_contact_capacity": 0.0,
    },
)
LOCAL_PENALTY = Rule(
    "Per-contact penalties (LocalMaxRayleighQuotient), the code default since 2026-11-06 (maintainer; the paper files "
    "have the global one): one global penalty, set by the stiffest contact, also stiffened soft-side contacts and "
    "slowed the sweeps. Global -> local, outer gradient ratio: Celebration median AAAVBD 40 -> 21, Newton 302 -> 20; "
    "BasketOfFruit 2 s median/max AAAVBD 117/1241 -> 51/306, VBD 270/2462 -> 41/220; DragonImpact impact p90/max "
    "AAAVBD 13.7/58.5 -> 6.3/15.0; CapsulesIntoBowl early exits 93% -> 100% and no feasibility violations; others "
    "within noise. Median feasibility ratios rise 2-5x where contact matters, staying below 1 (measurements: "
    "gpu/contact/dynamics.py, at adapt_penalty_parameters_to_hessian).",
    {"Contact.ContactDynamics.penalty_stiffness": 1},
)
STATIONARITY = Rule(
    "Stationarity tolerances rstol = 2e-2 and astol = 0.1, the code defaults since 2026-11-06 (maintainer; the paper "
    "files' rstol 1e-2 and astol 0.5 had former meanings): the mass-weighted RMS of each node's residual acceleration "
    "over the largest of rstol times its largest force per mass, astol times its own external acceleration and its "
    "float32 floor must be at most 1. "
    "rstol: Trampoline.SoftSoft and CapsulesIntoBowl traces gave the same pass points for 1e-2 to 1e-1. astol 0.1 "
    "(about 1 m/s^2): 2-4x closer to an early-exit-off reference than 0.5 at h = 10 ms for 35-48% more sweeps, and "
    "no effect at h = 1 ms (gpu/vbd/convergence.py, ConvergenceParams). The norm over nodes is the mass-weighted "
    "RMS (node_norm 0, the code default); the max (1) is stricter, as single nodes at their float32 floor can hold "
    "a step (Trampoline.SoftSoft passed 0 of 421 steps under a max test). Residual smoothing in the test, "
    "smoothing_passes = 4 with smoothing_in_test = 1, the code default for every scene since 2026-11-08 "
    "(maintainer): the test must also pass with the residual and its float32 floor smoothed by 4 passes of 1-ring "
    "averaging. It rejects unconverged states that the unsmoothed test passes through its floor clause (RMS_M(q0) > 1 "
    "but RMS_M(r) <= kg RMS_M(phi)), and it can only remove passes. Screen (2026-11-08, one step per solver from a "
    "shared AAAVBD checkpoint at the scene budget, early exit off; decisive ratio at k = 0 / k = 4 and mass-weighted "
    "RMS distance to the reference; data/profiling/investigations/smoothing-screen): BendingBeam.100MPa Chebyshev "
    "0.82 / 6.72, 0.43 mm from Newton; BendingBeam.Heterogeneous VBD, Chebyshev and Anderson 0.46-0.83 / 1.96-3.55, "
    "0.11-0.19 mm vs AAAVBD's 0.058 mm; BirthdayBanner VBD and Anderson 0.86 / 10.9 and 1.00 / 11.6, 3.5 mm from "
    "AAAVBD at 100x the budget, which fails k = 4 too (1.51), so the banner runs its full budget (accepted); "
    "BasketOfFruit AAAVBD at 10x the budget 0.92 / 15.1, 1.8 mm from 100x. Celebration (2026-11-07, 5 x 80; "
    "data/profiling/investigations/residual-smoothing): Chebyshev and Anderson pass unsmoothed (0.84 / 0.78) while "
    "their letters sag, 17 / 41 mm from Newton vs AAAVBD's 3.3 mm, and fail smoothed (12.5 / 10.7); "
    "BendingBeam.100MPa VBD (2026-11-07) passed steps 0-2 with a gravity-sized smooth error at k = 0, none at k = 4 "
    "(data/profiling/investigations/floor-clause-false-positive). Where the "
    "unsmoothed ratio only misranks the solvers (BendingBeam.10MPa, the trampolines, MatOnStilts, BasketOfFruit: "
    "AAAVBD the closest with the largest unsmoothed ratio), the smoothed ratio ranks them; every status row records "
    "both (unsmoothed_ratio, smoothed_ratio). Cost about +1.3% per step at a 20-sweep check interval (Celebration, "
    "0.22 sweep per test).",
    {"Solver.Convergence.rstol": 2e-2, "Solver.Convergence.astol": 0.1, "Solver.Convergence.node_norm": 0,
     "Solver.Convergence.smoothing_passes": 4, "Solver.Convergence.smoothing_in_test": 1},
)
PENALTY_FLOOR = Rule(
    "Contact penalty floor stiffness min_sigma_n = 1e4 N/m, the code default (maintainer, 2026-10-26): the effective "
    "normal penalty is floored at h2 * min_sigma_n (h2 the substep's BetaTilde^2), 1 at h = 10 ms with BDF1 like the "
    "former fixed floor of 1. The paper files' 1.0 is that former floor; read as a stiffness, it would floor the "
    "penalty at 1e-4 at 10 ms.",
    {"Contact.ContactDynamics.min_sigma_n": 1e4},
)


# Real-time frame steps. Each real-time entry is its 10 ms entry plus the frame step and its budget.
REALTIME_SCREEN = "data/profiling/investigations/RealtimeTimeSteps-screen"


def _realtime_step(hz: int) -> Rule:
    """The frame step dt = 1/``hz`` s, as the float32 value the screens used."""
    return Rule(
        "Real-time frame steps instead of 10 ms (maintainer, 2026-11-10): a 1/30 s frame first, then 1/60 s, and "
        "never a frame below 1/60 s. Substeps only as a last resort: when the sweeps needed at the full step would be "
        "prohibitive (caps of about 200 sweeps per solve for soft scenes and about 500 for heavy ones), or when the "
        "scene explodes because contact detection can't cope at that step. Each substep keeps the full per-solve "
        "budget, the same for every VBD-family solver; Newton keeps its 10 ms entry's outer x Newton iterations and "
        "CG cap per solve (untested at these steps). Screen (AAAVBD, early exit on): at the 10 ms budgets the "
        "dominant failure at larger steps is under-convergence (soft bodies sag and stretch, then contact scenes "
        "explode), not missed contacts; no scene keeps its 10 ms behaviour at 1/30 s without substeps, "
        "CapsulesIntoBowl comes closest (outcome kept, 23 mm overlaps). A 1/30 s frame with 2N substeps is bit for "
        "bit a 1/60 s frame with N. At 1/60 s, AAAVBD was the only VBD-family solver that kept the 10 ms outcome in "
        f"every scene (phase 2). Data: {REALTIME_SCREEN} (phases 1, 1b, 2 and 3, and FourBunnies without "
        "substeps).",
        {"Integration.dt": float(np.float32(1 / hz))},
    )


REALTIME_60HZ = _realtime_step(60)
REALTIME_30HZ = _realtime_step(30)


def _realtime_budget(substeps: int, outer: int, sweeps: int, why: str) -> Rule:
    """``substeps`` per frame step, each of ``outer`` x ``sweeps`` for every VBD-family solver; Newton keeps its own."""
    steps = "no substeps" if substeps == 1 else f"{substeps} substeps per frame step"
    return Rule(
        f"Real-time budget (maintainer, 2026-11-10): {steps}, {outer} outer iteration{'s' if outer > 1 else ''} x "
        f"{sweeps} sweeps = {outer * sweeps} sweeps per solve ({substeps * outer * sweeps} per frame step) for every "
        f"VBD-family solver. {why}",
        {"Integration.substeps": substeps, **_per_solver("n_max_iters", outer),
         **_per_solver("n_subproblem_max_iters", sweeps)},
    )


SCENES = {
    # The six scene configs that predate the explicit form: their values are kept, plus the 2 mm band.
    "BasketOfFruit": Scene(
        "BasketOfFruit",
        "final_scenes/basket_of_fruit.h5 (Young's modulus 1e5 to 1e8)",
        [BASKET_BUDGET, *SOLVERS, *HET_CONTACT, BASKET_MARGIN, PENALTY_FLOOR, LOCAL_PENALTY, STATIONARITY,
         CONTACT_CAPACITIES],
    ),
    "Celebration": Scene(
        "Celebration",
        "final_scenes/celebration.h5 (Young's modulus 1e6 to 1e8)",
        [
            _stiff_budget("1 x 5 x 30 = 150 sweeps"),
            *SOLVERS,
            *HET_CONTACT,
            Rule(
                "Contact for the celebration scene (maintainer, 2026-11-01): a 5 mm AL margin dmin and a 3 mm "
                "VertexSdf band, with the source's 5 cm max_dist (was 30 cm): 5 cm still covers any vertex inside "
                "the base plate (10.7 cm thick) or a confetto (about 5 cm thick).",
                {"Contact.CDType.VertexSdf.max_dist": 0.05, "Contact.CDType.VertexSdf.band": 3e-3,
                 "Contact.ContactDynamics.dmin": 5e-3},
            ),
            PENALTY_FLOOR, LOCAL_PENALTY, STATIONARITY,
            CONTACT_CAPACITIES,
            Rule(
                "Early exit for every solver, the VBD family and Newton (maintainer, 2026-11-07; Newton follows the "
                "shared early_exit), replacing the fixed budgets of 2026-11-01 (with residual smoothing in the "
                "test, STATIONARITY).",
                {"Solver.Convergence.early_exit": 1},
            ),
            Rule(
                "Stationarity tolerances 4x the code defaults (rstol 2e-2, astol 0.1; STATIONARITY) for every "
                "solver, Newton included (the convergence parameters are shared on purpose): rstol 0.08, astol 0.4 "
                "(maintainer, 2026-11-08; the defaults stay, tolerances are loosened per scene where needed). With "
                "the floor clause failing, a step passes at s times the tolerances iff s >= its smoothed RMS q0; at "
                "the end of the 400-sweep budget AAAVBD needs "
                "3.1-3.2x, Chebyshev about 26x, Anderson 22-43x, VBD about 50x. At 4x, 97% of AAAVBD's steps exit at "
                "sweep 341 of 400 (-14% sweeps, -13% time per step); its letters end 4.3 mm from Newton (3.3 mm at "
                "the full budget; Chebyshev 35, Anderson 80, VBD 118 mm), 1.3 mm median from the full-budget run, "
                "steady, with no drift; one impact step ends with feasibility ratio 1.30. 5x and 10x were "
                "bit-identical to 4x: the unsmoothed test passes only through its float32-floor clause, which doesn't "
                "scale with the tolerances. Data: data/profiling/investigations/celebration-tolerance-scaling.",
                {"Solver.Convergence.rstol": 8e-2, "Solver.Convergence.astol": 0.4},
            ),
            Rule(
                "Newton, the reference, with one-sided contact at 10 outer iterations x 2 Newton steps, CG rtol 1e-7 "
                "at most 6000 iterations (maintainer, 2026-11-01). With the slack fixed per Newton step, unpressed "
                "contacts were stiff two-sided springs that held confetti on the letters' vertical faces (5 x 1: none "
                "of the 7 on walls moved 5 cm in 0.2 s from 2 s); one-sided, 4 of 7 fall, one at free-fall speed. "
                "A second Newton step per subproblem brought the worst feasibility ratio from 3.06 back to 0.86 (a "
                "third: 0.24, the same confetti motion). At 2000 iterations CG stopped at its cap in 86-95% of the "
                "subproblems; it reaches rtol 1e-6 within 3910 iterations and 1e-7 within 4690 (2 steps from 2 s). "
                "About 65 s per step. Before (5 x 1, rtol 1e-6, 2000): 3 x 1 penetrates (feasibility ratio 1.4), and "
                "5 x 1 landed within the run-to-run noise of 10 x 1 on 6 impact steps.",
                {"Solver.Newton.one_sided_contact": 1, "Solver.Newton.n_max_iters": 10,
                 "Solver.Newton.newton.n_max_iters": 2, "Solver.Newton.LinearSolver.rtol": 1e-7,
                 "Solver.Newton.LinearSolver.n_max_iters": 6000},
            ),
        ],
    ),
    "MatOnStilts": Scene(
        "MatOnStilts.5Substeps",
        "final_scenes/mat_on_stilts_3.h5 (Young's modulus 1e6 to 1e8)",
        [
            _stiff_budget(
                "5 substeps of 5 x 5 = 125 sweeps; with them, EVD and gammaup/gammaupS 0.9/0.85, AAAVBD blows up "
                "once the mat lands on the stilts. 10 x 20 and 10 x 40 were also stable"
            ),
            *SOLVERS,
            *HET_CONTACT,
            PENALTY_FLOOR, LOCAL_PENALTY, STATIONARITY,
            CONTACT_CAPACITIES,
        ],
        ["The paper's 10-substep file (MatOnStilts.10Substeps) differs from the 5-substep one only in "
         "substeps, which this config overrides."],
    ),
    "Trampoline.SoftStiff": Scene(
        "Trampoline.SoftStiff.5Substeps",
        "final_scenes/trampolines/trampoline_{circle,square}_soft-stiff[__with_sphere].h5 (Young's modulus 1e6 to 1e8)",
        [_stiff_budget("5 substeps of 1 x 100 sweeps"), *SOLVERS, *HET_CONTACT, *TRAMPOLINE,
         PENALTY_FLOOR, LOCAL_PENALTY, STATIONARITY, CONTACT_CAPACITIES],
    ),
    "Trampoline.StiffSoft": Scene(
        "Trampoline.StiffSoft.5Substeps",
        "final_scenes/trampolines/trampoline_{circle,square}_stiff-soft[__with_sphere].h5 (Young's modulus 1e6 to 1e8)",
        [_stiff_budget("5 substeps of 1 x 25 AAAVBD sweeps, 1 x 100 for VBD"), *SOLVERS, *HET_CONTACT,
         *TRAMPOLINE, PENALTY_FLOOR, LOCAL_PENALTY, STATIONARITY, CONTACT_CAPACITIES],
    ),
    "Trampoline.SoftSoft": Scene(
        "Trampoline.SoftStiff.5Substeps",
        "final_scenes/trampolines/trampoline_{circle,square}_soft-soft[__with_sphere].h5 (homogeneous, Young's "
        "modulus 1e6)",
        [
            _budget(20, "Homogeneous and soft, about 100 sweeps per step by the maintainer's guidance",
                    "5 substeps of 1 x 100 sweeps, for the soft-stiff trampolines"),
            *SOLVERS,
            *CONTACT,
            *TRAMPOLINE,
            PENALTY_FLOOR, LOCAL_PENALTY, STATIONARITY,
            CONTACT_CAPACITIES,
        ],
        ["The paper has no parameter file for the soft-soft trampolines; the source is the soft-stiff one's."],
    ),
    "FourBunnies.Trampoline250kgPerM3": Scene(
        "FourBunnies",
        "data/scenarios/FourBunnies.Trampoline250kgPerM3.h5: accelerated-vbd-extras final_scenes/4bunny_bounce.h5 "
        "with the trampoline's density lowered from 1000 to 250 kg/m^3 (maintainer, 2026-11-09), everything else "
        "unchanged: a circular trampoline (radius 2 m, 10 cm thick) whose rim is pulled radially to 3 m over the first "
        "second and then held, and four bunnies (bunny.mesh, 1000 kg/m^3, 279 kg each) held 0.8 m above it until 1 s, "
        "then dropped; Young's modulus 1 MPa and Poisson's ratio 0.45 everywhere",
        [*SOLVERS, *CONTACT, *TRAMPOLINE, PENALTY_FLOOR, LOCAL_PENALTY, STATIONARITY, CONTACT_CAPACITIES,
         FOUR_BUNNIES_BDF2],
        ["Budget kept: the source's 5 x 20 = 100 sweeps per step, no substeps, as Trampoline.SoftSoft's (homogeneous "
         "and soft).",
         "The trampoline's 1 cm VertexSdf band (TRAMPOLINE) also covers the bunnies' 4 m/s impacts.",
         "Why the lighter trampoline (BDF1, AAAVBD, 0-4 s): trampoline 1000 / 250 / 100 kg/m^3 with the bunnies at "
         "1000 gave rebound ratios 0.33 / 0.41 / 0.42 and first bounces 18 / 24 / 25 cm clear of the trampoline (second "
         "bounce none / 11 / 16 cm), 250 with the fewest penetration violations (1 step, ratio 1.05). Heavier bunnies "
         "(2000 and 4000 kg/m^3) kept the ratio at 0.34, sank deeper (0.5 and 0.8 m) and barely left the trampoline "
         "(13 and 2 cm), with feasibility unmet on 56-122 steps; trampoline 250 with bunnies 4000 diverged at 2.88 s. "
         "Data: data/profiling/investigations/FourBunnies-density-tuning."],
    ),
    # The other paper configs: the durable rules only, budgets unchanged
    "Trampoline.StiffSphere": Scene(
        "Trampoline.StiffSphere.5Substeps",
        "data/scenarios/Trampoline.Square.Soft.StiffSphere.h5, converted from final_scenes/trampoline_update.zip "
        "(trampoline_soft_sphere_stiff.Dump.h5): the square 1 MPa trampoline with a 100 MPa sphere",
        [*SOLVERS, *HET_CONTACT, *TRAMPOLINE, PENALTY_FLOOR, LOCAL_PENALTY, STATIONARITY, CONTACT_CAPACITIES,
         Rule("One budget for every VBD-family solver, AAAVBD's 1 x 100 sweeps per substep (maintainer, "
              "2026-11-09: the paper compares the solvers at equal cost; the paper file gives 1 x 400 to VBD).",
              _per_solver("n_subproblem_max_iters", 100))],
        ["Substeps kept: 5 substeps of 1 x 100 sweeps, one contact detection pass per substep. The other "
         "trampoline scene configs use 1 substep of 5 x 80 instead."],
    ),
    "CapsulesIntoBowl": Scene(
        "CapsulesIntoBowl",
        "final_scenes/capsule_bowl.h5 and sessions/capsule_bowl.h5 (tuned on the latter; Young's modulus 1e5 to 1e6; "
        "the bodies span about 8.8 x 8.8 x 8.5 m)",
        [*SOLVERS, PENALTY, CAPSULE_CONTACT, CAPSULE_BUDGET, PENALTY_FLOOR, LOCAL_PENALTY, STATIONARITY, CONTACT_CAPACITIES],
    ),
    "ArmadilloSpaghetti.DragonImpact": Scene(
        "defaults",
        "data/scenarios/ArmadilloSpaghetti.DragonImpact.h5: four round pillars, cylinders 0.1 m in diameter "
        "(fTetWild, ideal edge 1.2 cm, mean 1.1 cm; 207k tets each, quality at least 0.10; Young's modulus 1e7) 0.52 m "
        "from the twist axis (0.735 m apart), their top and bottom ends turned at -180 and 180 deg/s from 0 s, between "
        "a floor and a ceiling plane 2 cm beyond the pillars' ends, around one soft armadillo (data/armadillo.mesh, "
        "162k tets, 1.2 m tall, Young's modulus 1e6; tet quality at least 0.23, triangle angles at least 11 deg) whose "
        "waist (the slice at 50-55% of its height, where only the torso is) is near the twist axis: lowered 15 cm "
        "along it (z from -0.75 to 0.45 m; floor 1.07 m below), then turned -10 deg about it and moved (+10.0, -2.5) "
        "cm in xy (pose A, maintainer, 2026-10-23), with rest gaps of 4.8, 7.2, 7.7 and 2.6 cm to the four pillars, "
        "so that all four pillars engage it. Thin pillars close to the armadillo wrap it rather than push it out "
        "(after the JGS2-GQ paper's scene). Gravity (-9.81 m/s^2 in z) acts on every body. The armadillo's interior "
        "core (its 1031 nodes within 25 cm of its centroid and at least 2 cm under its surface) is held by a Fixed "
        "transform until 1.5 s, so that the twisting pillars tangle around it rather than push it up or down, and "
        "then released (maintainer, 2026-10-22). The held nodes stay off the surface: holding surface nodes pits the "
        "pillars' prescribed twist against prescribed armadillo nodes in contact, with no feasible position, and the "
        "simulation froze. The twist stops at 4.0 s (720 deg per end; maintainer, 2026-10-24), and a Fixed "
        "transform on the pillars' turned end nodes over [0, 1000) s keeps the ends where the rotation stopped. "
        "Two Stanford dragons (data/stanford-dragon_ftetwild.msh scaled to 1.5 m long, 51k nodes and 204k tets "
        "each, Young's modulus 1e7, Poisson's ratio 0.45, 1000 kg/m^3) parked left (-x) and right (+x) of the twist "
        "axis, both leading with the mesh's higher, narrower +x end (the right one turned 180 deg about z): the left "
        "one's at the armadillo's feet (z -0.60 m), the right one's at its shoulders (z +0.25 m), both at y 1.45 m and "
        "1.1 and 1.3 m from the axis. Each is held by a Fixed transform over [0, 4.49) s and launched by a Translate "
        "over [4.49, 4.50) s at 5 m/s toward the axis plus the upward speed (0.62 and 0.89 m/s) that brings its "
        "leading end back to its launch height over the flight to the bodies' near side (0.13 and 0.18 s, aimed at "
        "the bodies' positions at 4.49 s in a run of ArmadilloSpaghetti.Fall.h5, the twist without its stop); "
        "released, it keeps that velocity. 1.40M tets in all",
        [*SOLVERS, PENALTY, ARMADILLO_SPAGHETTI_CONTACT, DRAGON_IMPACT_BUDGET, PENALTY_FLOOR, LOCAL_PENALTY, STATIONARITY,
         CONTACT_CAPACITIES],
        ["Built from ArmadilloSpaghetti.Fall.h5 (left unchanged); the twist stop, the launch time and the aim are the "
         "scene file's transform windows and velocities. Its variant with the twist stopped at 5.0 s and the launch at "
         "5.49 s is data/scenarios/ArmadilloSpaghetti.DragonImpact.TwistStop5s.h5.",
         "Why 4.0 s (AAAVBD, 2026-10-24): with the twist stopped at 5.0 s (10 substeps of 1 x 100), the armadillo "
         "kept swinging (about 0.8 m/s RMS at 1.3 Hz, no decay from 8 to 20 s), and a cluster of inverted elements "
         "(about 19 within 20 cm of element 846743, which first inverts at about 3.8 s in the twist) released at about 18.9 s (the "
         "armadillo dropped 9 cm, penetration ratio 4.1, nodes at 28 m/s). With the twist stopped at 4.0 s (5 "
         "substeps of 1 x 80, launch to 4 s after it), the cluster stayed at 1-2 inverted elements and the pillars "
         "moved less (mid-span RMS 0.28-0.40 m/s, reversal 0.18-0.37)."],
    ),
    "ArmadilloPull": Scene(
        "defaults",
        "sessions/armadillo-pull_4.h5 (no contact: the armadillo pulled, then released)",
        [INVERSE, ARMADILLO_GAMMAS, ARMADILLO_BUDGET, STATIONARITY],
    ),
    "EmptyBasket": Scene(
        "EmptyBasket",
        "probably final_scenes/short_basket.h5 (the woven basket alone, homogeneous, Young's modulus 1e7; not "
        "recorded)",
        [*SOLVERS, *CONTACT, BASKET_MARGIN, PENALTY_FLOOR, LOCAL_PENALTY, STATIONARITY, CONTACT_CAPACITIES],
        ["Budgets kept: 3 substeps of 10 x 5 = 150 sweeps per frame, a little below the guidance's ~200 for "
         "Young's modulus ~1e7."],
    ),
    "BasketOfOranges": Scene(
        "BasketOfOranges",
        "a basket scene, probably final_scenes/basket_of_oranges.h5 (Young's modulus 1e6 to 1e8; not recorded)",
        [*SOLVERS, *HET_CONTACT, BASKET_MARGIN, PENALTY_FLOOR, LOCAL_PENALTY, STATIONARITY, CONTACT_CAPACITIES],
        ["Budgets kept: 5 substeps of 5 x 40 = 1000 sweeps per frame."],
    ),
    # Contact-free scenes (VertexSdf.deactivate = 1): the contact rules don't apply (only the margin, for consistency),
    # Newton keeps its budget
    "BirthdayBanner": Scene(
        "BirthdayBanner",
        "the birthday banner (scene file not in final_scenes; tetrahedra/banner_regions.mesh)",
        [*SOLVERS, DMIN, STATIONARITY],
        ["Contact detection is off in the source (VertexSdf.deactivate = 1), so its contact parameters are kept.",
         "The positive margin (DMIN) is set for consistency with the other scenes; it has no effect while VertexSdf "
         "detection is off.",
         "Budgets kept: 1 x 2000 sweeps per step."],
    ),
    **{
        f"BendingBeam.{name}": Scene(
            f"BendingBeam.{name}.GroundTruth",
            f"final_scenes/bending_beams/small_beam{'_het' if y == '_het' else 'Y' + y}.h5 ("
            + ("Young's modulus 1e6 to 1e8" if y == "_het" else f"homogeneous, Young's modulus {y}")
            + ")",
            [*SOLVERS, DMIN, STATIONARITY],
            ["Contact-free (single body, VertexSdf.deactivate = 1): contact parameters kept.",
             "The positive margin (DMIN) is set for consistency with the other scenes; it has no effect while "
             "VertexSdf detection is off.",
             "Budgets kept: the paper's ground-truth budgets, chosen to match Newton's behaviour "
             "(accelerated-vbd-extras/parameters/gpu/beam.GroundTruth.Notes.txt)."]
            + (["The paper's BendingBeam.10MPa.GroundTruth.5Substeps (5 substeps of 20 AAAVBD / 500 VBD sweeps) is a variant "
                "of this scene."] if y == "1e7" else []),
        )
        for y, name in (("1e5", "100kPa"), ("1e6", "1MPa"), ("1e7", "10MPa"), ("1e8", "100MPa"),
                        ("_het", "Heterogeneous"))
    },
}


# DragonImpact at the lower budget: the same scene file, its own config
_DRAGON_IMPACT = SCENES["ArmadilloSpaghetti.DragonImpact"]
SCENES["ArmadilloSpaghetti.DragonImpact.Fast"] = Scene(
    _DRAGON_IMPACT.source,
    _DRAGON_IMPACT.scene_files,
    [*_DRAGON_IMPACT.rules, DRAGON_IMPACT_FAST_BUDGET],
    ["For AAAVBD. VBD fails at this budget from t = 0 with every multiplier step tried: during the twist (step 84) "
     "with 0.5-0.9, at the rebound (step 579) with 0.25; use ArmadilloSpaghetti.DragonImpact for VBD. Newton keeps "
     "ArmadilloSpaghetti.DragonImpact's budget.",
     *_DRAGON_IMPACT.notes],
)


def _realtime(base: str, step: Rule, budget: Rule, notes=()) -> Scene:
    """``base``'s 10 ms entry at a real-time frame ``step`` with its ``budget``."""
    scene = SCENES[base]
    return Scene(
        scene.source,
        scene.scene_files,
        [*scene.rules, step, budget],
        [*notes, f"Every other setting is {base}'s (10 ms); its rules above keep their 10 ms wording, and their "
         "per-solve settings apply per substep.", *(f"10 ms entry: {note}" for note in scene.notes)],
    )


# Real-time entries, next to their 10 ms entries until those are removed (maintainer, 2026-11-10). The evidence quoted
# is AAAVBD's, against its 10 ms run (REALTIME_SCREEN).
SCENES.update({
    "CapsulesIntoBowl.60Hz": _realtime("CapsulesIntoBowl", REALTIME_60HZ, _realtime_budget(
        1, 10, 20,
        "1/30 s already runs the 200 cap and keeps the outcome, but with contact worse than at 10 ms "
        "(CapsulesIntoBowl.30Hz). 1/60 s is as clean as 10 ms: interpenetration at most 10 mm (dmin 10 mm; 10 ms "
        "10.1 mm), feasibility max 0.23, stationarity ratio median 0.61 (smoothed 0.04), 100% early exits, 5.6 GPU s "
        "per simulated s (10 ms 6.7).")),
    "CapsulesIntoBowl.30Hz": _realtime("CapsulesIntoBowl", REALTIME_30HZ, _realtime_budget(
        1, 10, 20,
        "A showcase of the large step (maintainer, 2026-11-10), next to the rule's choice CapsulesIntoBowl.60Hz. "
        "AAAVBD keeps the outcome (all 26 capsules in the bowl) but leaves contact overlaps up to 23 mm (dmin 10 mm; "
        "12 frames with a vertex more than 1 cm inside a partner, feasibility > 1 on 71 solves, max 6.6; 10 ms: 1 "
        "frame and 0 solves), from contact at this step rather than the budget (a 10x-sweep reference still reaches "
        "feasibility 21). At the same budget the other solvers leave capsules stuck in the tubes: VBD 21 of 26, "
        "Chebyshev 7, Anderson 2 (none at 10 ms or 1/60 s), with overlaps of 44, 40 and 69 mm.")),
    "Celebration.60Hz": _realtime("Celebration", REALTIME_60HZ, _realtime_budget(
        1, 5, 80,
        "1/30 s blows up as the confetti land (0.47-0.53 s, contact overflow) at 5 x 80 and at 5 x 160. 1/60 s: no "
        "interpenetration, feasibility max 1.4 (> 1 on 4 solves; 10 ms 1.3, 1), stationarity ratio median 2.1 "
        "(smoothed 0.86; 10 ms 0.90 (0.81)), 0% early exits (10 ms 97%), 47 GPU s per simulated s (10 ms 68).")),
    "MatOnStilts.60Hz": _realtime("MatOnStilts", REALTIME_60HZ, _realtime_budget(
        1, 5, 160,
        "At 1/30 s and 5 x 80 the mat drapes over the stilt like a cloth. At 1/60 s, 5 x 80 and 5 x 120 change the "
        "outcome (the cow slides off first; at 5 x 120 the mat tips about 0.5 s later). At 5 x 160 the mat tips off "
        "with the cow as at 10 ms, 0.2-0.3 s later: no interpenetration, feasibility max 2.0 (> 1 on 4 solves; 10 ms "
        "4.9, 4), stationarity ratio median 13.0 (smoothed 3.6; 10 ms 7.8 (2.9)), 19% early exits (10 ms 16%), 11.5 "
        "GPU s per simulated s. 800 sweeps per step is above the ~500 cap, accepted by the maintainer: it costs the "
        "same as 2 substeps of 5 x 80 (11.7).")),
    "FourBunnies.Trampoline250kgPerM3.60Hz": _realtime("FourBunnies.Trampoline250kgPerM3", REALTIME_60HZ,
                                                       _realtime_budget(
        1, 5, 60,
        "1/30 s blows up at the first impact (1.43 s), at 5 x 20 and 5 x 60 and with a 14 cm band. At 1/60 s the 10 "
        "ms budget, 5 x 20, loses a bounce (second apex 0.49-0.59 m vs 0.84-0.89 m) with 11.6 mm interpenetration "
        "(dmin 1 mm). At 5 x 60 the bounces match 10 ms within 3 cm (0.98-0.99 and 0.83-0.89 m), all four bunnies "
        "stay on the trampoline for 10 s, interpenetration at most 6.3 mm with no vertex more than 1 cm inside "
        "another body, 9.7 GPU s per simulated s (3 substeps of 5 x 20: 10.9). More outer iterations (15 x 20) lost "
        "a bunny off the edge at 5.68 s; a 7 cm band alone still lost the second bounce.")),
    "BirthdayBanner.60Hz": _realtime("BirthdayBanner", REALTIME_60HZ, _realtime_budget(
        2, 1, 2000,
        "Substeps: with 1 substep the banner goes NaN at 1/30 s (1.0 s, stretched to 3.7 m) and is far too soft at "
        "1/60 s (centroid -0.79 m at 0.5 s vs -0.57 m). With 2 (h = 1/120 s) it sags and swings as at 10 ms "
        "(centroid -0.51 m at 0.5 s, 5 cm from 10 ms at 8 s), stationarity ratio median 8.7 (10 ms 12.6), 71.6 GPU "
        "s per simulated s (10 ms 58.9). 1 substep of 1 x 4000, at equal cost, followed 10 ms about as closely but "
        "was less converged (ratio 17.2 vs 8.0 over 0-2 s).")),
    "ArmadilloPull.60Hz": _realtime("ArmadilloPull", REALTIME_60HZ, _realtime_budget(
        2, 1, 1000,
        "Substeps: with 1 substep the armadillo goes NaN at 1/60 s right after the release (1.05-1.08 s), even with "
        "10x the sweeps, and is far too soft at 1/30 s (stationarity ratio 31 vs 1.4). With 2 (h = 1/120 s) it stays "
        "finite through the release with the same swing and final pose (centroid 0.15 m from 10 ms at 8 s), "
        "stationarity ratio median 0.98 (smoothed 0.26), 77% early exits (10 ms 28%), 26.6 GPU s per simulated s "
        "(10 ms 25.0).")),
    "Trampoline.StiffSphere.60Hz": _realtime("Trampoline.StiffSphere", REALTIME_60HZ, _realtime_budget(
        5, 1, 100,
        "Substeps: with 1 substep the scene blows up (1/30 s at 1 x 200 and 1 x 500, 1/60 s at 1 x 200), and at "
        "1/60 s and 1 x 500 the sphere sinks to -1.84 m (10 ms -0.65 m) with feasibility > 1 on 122 solves. 5 "
        "substeps at the config's 1 x 100 are cheaper and tighter than 2 at the cap (maintainer, 2026-11-10): "
        "bounces within about 2 cm of 10 ms (1.34 vs 1.32 m), feasibility > 1 on 14 solves (10 ms 9), 53% early "
        "exits, 7.3 GPU s per simulated s, against 2 substeps of 1 x 500: feasibility > 1 on 32, 13.0 GPU s per "
        "simulated s.")),
    "BasketOfFruit.60Hz": _realtime("BasketOfFruit", REALTIME_60HZ, _realtime_budget(
        5, 3, 50,
        "Substeps: with 1 substep the scene blows up (1/60 s at 3 x 70 and 3 x 170, contact overflow), so 1/30 s "
        "wasn't tried. 5 substeps at the config's 3 x 50 are cheaper and "
        "tighter than 2 at the cap (maintainer, 2026-11-10): every fruit stays in the basket, no interpenetration, "
        "feasibility max 3.1 (> 1 on 8 solves), 292 GPU s per simulated s (10 ms 521), against 2 substeps of 3 x 170: "
        "max 4.5 (> 1 on 66 solves over 2 s), 373-381 GPU s per simulated s.")),
    "ArmadilloSpaghetti.DragonImpact.60Hz": _realtime("ArmadilloSpaghetti.DragonImpact", REALTIME_60HZ,
                                                      _realtime_budget(
        10, 1, 100,
        "Substeps: 1-3 substeps explode at the impact (at 1 x 500: 1 substep at 4.65 s, 2 at 4.15 s as the pillars "
        "spring back when their twist stops, 3 at 4.25 s); 4-9 at 1 x 500 would cost 2-4.5x the 10 substeps of 1 x "
        "100. With 10 (h = 1/600 s): the same outcome as 10 ms, no interpenetration, feasibility > 1 on 8 solves "
        "(10 ms 612), 242 GPU s per simulated s (10 ms 394); the dragons' rebound is 1.1 m from 10 ms at 6 s.")),
})

SOFT_STIFF_BDF2 = Rule(
    "BDF2 time integration for the sphere's bounce at 1/60 s (maintainer, 2026-11-11): BDF1 damps it numerically. "
    "Bounce heights of the sphere (AAAVBD, 5 x 80, 0-6 s) against a small-step reference, 1/60 s x 10 substeps (h = "
    "1/600 s) with BDF2, whose bounce is nearly elastic (1.70 then 1.71 m; BDF1 at that step 1.64 then 1.43 m): "
    "1/60 s with BDF2 1.58 and 1.12 m, mean / max RMS distance to the reference 195 / 599 mm, 6.1 GPU s per "
    "simulated s; 2 substeps with BDF1 1.34 and 0.88 m, 244 / 1144 mm, 13.5; 1/60 s with BDF1 0.79 and 0.64 m, 355 "
    f"/ 1257 mm; 10 ms with BDF1 1.36 and 0.82 m, 243 / 1107 mm. Data: {REALTIME_SCREEN}.",
    {"Integration.bdf_scheme": 2},
)

SCENES["Trampoline.SoftStiff.60Hz"] = _realtime("Trampoline.SoftStiff", REALTIME_60HZ, _realtime_budget(
    1, 5, 80,
    "1/30 s at 5 x 160 sags the trampoline (0.19-0.30 m vs 0.14-0.19 m at 10 ms) and loses the bounce (0.52 vs 1.34 "
    "m), with feasibility > 1 on 12 solves. 1/60 s without substeps at the 10 ms budget: no interpenetration, "
    "feasibility max 1.1 (> 1 on 1 solve), stationarity ratio median 15.1 (smoothed 6.0), as at 10 ms (15.0); "
    "5 x 100 (the cap) didn't restore the BDF1 bounce (0.86 m), a smaller step or BDF2 does."))
SCENES["Trampoline.SoftStiff.60Hz"].rules.append(SOFT_STIFF_BDF2)


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _nest(flat: dict) -> dict:
    out = {}
    for key, v in flat.items():
        *tables, k = key.split(".")
        d = out
        for t in tables:
            d = d.setdefault(t, {})
        d[k] = v
    return out


def _same(a, b) -> bool:
    """Equal as the float32 / int the simulator stores, so 0.99 matches a converted 0.9900000095367432."""
    if isinstance(a, float) or isinstance(b, float):
        return np.float32(a) == np.float32(b)
    return a == b


def _read(path: pathlib.Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def build(name: str) -> tuple[dict, list[str]]:
    """The scene's full config and its header."""
    scene = SCENES[name]
    source_path = CONFIGS / "defaults.toml" if scene.source == "defaults" else CONFIGS / "paper" / f"{scene.source}.toml"
    defaults = _flatten(_read(CONFIGS / "defaults.toml"))
    paper = _flatten(config.read(source_path))
    # Parameters added since the paper configs were generated keep their code defaults (the paper configs
    # aren't regenerated: they'll be replaced by these)
    from_defaults = sorted(defaults.keys() - paper.keys())
    source = {**{k: defaults[k] for k in from_defaults}, **paper}
    keys = set(defaults)
    cfg, why = dict(source), {}
    for rule in scene.rules:
        for key, value in rule.values.items():
            if key not in keys:
                raise KeyError(f"{name}: {key} is not a serialized parameter")
            cfg[key] = source[key] if _same(source[key], value) else value
            why[key] = rule
    changed = [k for k in cfg if not _same(cfg[k], source[k])]
    header = [
        f"Scene config: {name}, every serialized parameter written out (diff it against configs/paper).",
        f"For: {scene.scene_files}.",
        f"Source: configs/{source_path.relative_to(CONFIGS).as_posix()}. Keys not listed below have the source's values.",
        f"Generated {datetime.date.today().isoformat()} by python/tools/sim/scene_configs.py; edit SCENES there, "
        "not this file.",
    ]
    rules = [r for r in scene.rules if any(why[k] is r for k in changed)]
    if rules:
        header += ["", "Differs from the source:"]
    for rule in rules:
        header += textwrap.wrap(rule.reason, 110, initial_indent="- ", subsequent_indent="  ", break_on_hyphens=False)
        header += [f"    {k} = {cfg[k]!r} (source {source[k]!r})" for k in changed if why[k] is rule]
    if from_defaults:
        header += ["", "Not in the source, code defaults (configs/defaults.toml):"] + [f"  {k}" for k in from_defaults]
    if scene.notes:
        header += ["", "Notes:"]
        for note in scene.notes:
            header += textwrap.wrap(note, 110, initial_indent="- ", subsequent_indent="  ", break_on_hyphens=False)
    return _nest({k: cfg[k] for k in sorted(cfg)}), header


def make() -> None:
    (CONFIGS / "scenes").mkdir(exist_ok=True)
    for name in SCENES:
        cfg, header = build(name)
        (CONFIGS / "scenes" / f"{name}.toml").write_text(config.dumps(cfg, header), encoding="utf-8")
        print(f"{name}.toml: {sum(line.startswith('    ') for line in header)} keys differ from {SCENES[name].source}")


def check() -> int:
    """Number of scene configs that differ from what :func:`make` writes (0 when all pass)."""
    bad = 0
    for name in SCENES:
        path = CONFIGS / "scenes" / f"{name}.toml"
        expected = _flatten(build(name)[0])
        actual = _flatten(_read(path)) if path.exists() else {}
        problems = [f"{k}: {actual.get(k)!r}, expected {v!r}" for k, v in expected.items() if actual.get(k) != v]
        problems += [f"unexpected key {k}" for k in actual.keys() - expected.keys()]
        bad += bool(problems)
        print(f"{name}: {'OK' if not problems else 'MISMATCH'}" + "".join(f"\n  {p}" for p in problems))
    stray = sorted(p.stem for p in (CONFIGS / "scenes").glob("*.toml") if p.stem not in SCENES)
    if stray:
        print(f"not generated here: {stray}")
        bad += len(stray)
    print(f"{bad} mismatching config(s)")
    return bad


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["make", "check"])
    args = parser.parse_args()
    if args.command == "make":
        make()
    else:
        sys.exit(check())


if __name__ == "__main__":
    main()
