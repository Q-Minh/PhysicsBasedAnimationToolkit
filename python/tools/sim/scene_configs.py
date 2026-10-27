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
    "2026-10-27): 40 sweeps per substep for every VBD-family solver, with half the AL multiplier step. From the "
    "100-sweep runs' 4.4 s checkpoints: AAAVBD diverged at step 495 with the standard step, and with half of it ran to "
    "18 s at about 0.49x the 100-sweep step time, its pillars and armadillo moving 1.4x and 1.25x as fast as at 100 "
    "sweeps at 17-18 s. VBD ran to 18 s with either step; half of it lengthened the rebound and raised the pillars' "
    "late motion (0.31 against 0.19 m/s), but multiplier_step is a contact parameter, shared by every solver "
    "(docs/solver-accuracy-notes.md, entries 12 and 14).",
    {**_per_solver("n_subproblem_max_iters", 40), "Contact.ContactDynamics.multiplier_step": 0.5},
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
PENALTY_FLOOR = Rule(
    "Contact penalty floor stiffness min_sigma_n = 1e4 N/m, the code default (maintainer, 2026-10-26): the effective "
    "normal penalty is floored at h2 * min_sigma_n (h2 the substep's BetaTilde^2), 1 at h = 10 ms with BDF1 like the "
    "former fixed floor of 1. The paper files' 1.0 is that former floor; read as a stiffness, it would floor the "
    "penalty at 1e-4 at 10 ms.",
    {"Contact.ContactDynamics.min_sigma_n": 1e4},
)

SCENES = {
    # The six scene configs that predate the explicit form: their values are kept, plus the 2 mm band.
    "BasketOfFruit": Scene(
        "BasketOfFruit",
        "final_scenes/basket_of_fruit.h5 (Young's modulus 1e5 to 1e8)",
        [_stiff_budget("3 substeps of 10 x 15 = 450 sweeps"), *SOLVERS, *HET_CONTACT, PENALTY_FLOOR,
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
                "max_dist 30 cm, scaled to the geometry (maintainer): the base plate is 10 cm thick and the "
                "letters' edges ~4 cm.",
                {"Contact.CDType.VertexSdf.max_dist": 0.3},
            ),
            PENALTY_FLOOR,
            CONTACT_CAPACITIES,
        ],
        ["The previous scene config's 1 cm band (for the 10 cm base plate) is now 2 mm, like every scene's."],
    ),
    # TODO: With this config (AAAVBD), the mat penetrates about 2 cm after its 2 s release, loses contact by about
    # 3.5 s and falls (z -26 m at 5 s), at dmin 1e-3 and -1e-4 and with the TIMESTEP penalty (2026-10-20). Next:
    # play the output to see whether it tunnels or slides off, compare with the 10-substep paper config, and
    # bisect from ff63102b/78232143, which introduced this config and changed it to 5 x 80.
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
            PENALTY_FLOOR,
            CONTACT_CAPACITIES,
        ],
        ["The paper's 10-substep file (MatOnStilts.10Substeps) differs from the 5-substep one only in "
         "substeps, which this config overrides."],
    ),
    "Trampoline.SoftStiff": Scene(
        "Trampoline.SoftStiff.5Substeps",
        "final_scenes/trampolines/trampoline_{circle,square}_soft-stiff[__with_sphere].h5 (Young's modulus 1e6 to 1e8)",
        [_stiff_budget("5 substeps of 1 x 100 sweeps"), *SOLVERS, *HET_CONTACT, *TRAMPOLINE,
         PENALTY_FLOOR, CONTACT_CAPACITIES],
    ),
    "Trampoline.StiffSoft": Scene(
        "Trampoline.StiffSoft.5Substeps",
        "final_scenes/trampolines/trampoline_{circle,square}_stiff-soft[__with_sphere].h5 (Young's modulus 1e6 to 1e8)",
        [_stiff_budget("5 substeps of 1 x 25 AAAVBD sweeps, 1 x 100 for VBD"), *SOLVERS, *HET_CONTACT,
         *TRAMPOLINE, PENALTY_FLOOR, CONTACT_CAPACITIES],
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
            PENALTY_FLOOR,
            CONTACT_CAPACITIES,
        ],
        ["The paper has no parameter file for the soft-soft trampolines; the source is the soft-stiff one's."],
    ),
    # The other paper configs: the durable rules only, budgets unchanged
    "Trampoline.StiffSphere": Scene(
        "Trampoline.StiffSphere.5Substeps",
        "the trampolines with a sphere (final_scenes/trampolines/*__with_sphere.h5; which variant is not recorded)",
        [*SOLVERS, *HET_CONTACT, *TRAMPOLINE, PENALTY_FLOOR, CONTACT_CAPACITIES],
        ["Budgets kept: 5 substeps of 1 x 100 AAAVBD sweeps (1 x 400 for VBD), one contact detection pass per "
         "substep. The other trampoline scene configs use 1 substep of 5 x 80 instead."],
    ),
    "CapsulesIntoBowl": Scene(
        "CapsulesIntoBowl",
        "final_scenes/capsule_bowl.h5 and sessions/capsule_bowl.h5 (tuned on the latter; Young's modulus 1e5 to 1e6; "
        "the bodies span about 8.8 x 8.8 x 8.5 m)",
        [*SOLVERS, PENALTY, CAPSULE_CONTACT, CAPSULE_BUDGET, PENALTY_FLOOR, CONTACT_CAPACITIES],
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
        [*SOLVERS, PENALTY, ARMADILLO_SPAGHETTI_CONTACT, DRAGON_IMPACT_BUDGET, PENALTY_FLOOR,
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
        [INVERSE, ARMADILLO_GAMMAS, ARMADILLO_BUDGET],
    ),
    "EmptyBasket": Scene(
        "EmptyBasket",
        "probably final_scenes/short_basket.h5 (the woven basket alone, homogeneous, Young's modulus 1e7; not "
        "recorded)",
        [*SOLVERS, *CONTACT, PENALTY_FLOOR, CONTACT_CAPACITIES],
        ["Budgets kept: 3 substeps of 10 x 5 = 150 sweeps per frame, a little below the guidance's ~200 for "
         "Young's modulus ~1e7."],
    ),
    "BasketOfOranges": Scene(
        "BasketOfOranges",
        "a basket scene, probably final_scenes/basket_of_oranges.h5 (Young's modulus 1e6 to 1e8; not recorded)",
        [*SOLVERS, *HET_CONTACT, PENALTY_FLOOR, CONTACT_CAPACITIES],
        ["Budgets kept: 5 substeps of 5 x 40 = 1000 sweeps per frame."],
    ),
    # Contact-free scenes (VertexSdf.deactivate = 1): the contact rules don't apply, Newton keeps its budget
    "BirthdayBanner": Scene(
        "BirthdayBanner",
        "the birthday banner (scene file not in final_scenes; tetrahedra/banner_regions.mesh)",
        [*SOLVERS],
        ["Contact detection is off in the source (VertexSdf.deactivate = 1), so its contact parameters are kept.",
         "Budgets kept: 1 x 2000 sweeps per step."],
    ),
    **{
        f"BendingBeam.{name}": Scene(
            f"BendingBeam.{name}.GroundTruth",
            f"final_scenes/bending_beams/small_beam{'_het' if y == '_het' else 'Y' + y}.h5 ("
            + ("Young's modulus 1e6 to 1e8" if y == "_het" else f"homogeneous, Young's modulus {y}")
            + ")",
            [*SOLVERS],
            ["Contact-free (single body, VertexSdf.deactivate = 1): contact parameters kept.",
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
    ["Untested from t = 0: the 40-sweep runs started from the 100-sweep runs' 4.4 s checkpoints, and at 20 sweeps "
     "both AAAVBD and VBD failed during the twist (solver notes, entry 9). Newton keeps "
     "ArmadilloSpaghetti.DragonImpact's budget.",
     *_DRAGON_IMPACT.notes],
)


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
    source = _flatten(config.read(source_path))
    keys = set(_flatten(_read(CONFIGS / "defaults.toml")))
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
    if scene.notes:
        header += ["", "Notes:"]
        for note in scene.notes:
            header += textwrap.wrap(note, 110, initial_indent="- ", subsequent_indent="  ", break_on_hyphens=False)
    missing = keys - set(cfg)
    if missing:
        raise KeyError(f"{name}: source lacks {sorted(missing)}; rerun paper_configs make")
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
