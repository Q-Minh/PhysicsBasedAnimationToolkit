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
    source: str  # configs/paper/<source>.toml
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
    "max_dist 30 cm, scaled to the metre-scale trampoline geometry (10 cm plate; maintainer).",
    {"Contact.CDType.VertexSdf.max_dist": 0.3},
)
SPHERE_BAND = Rule(
    "VertexSdf band 5 mm for the sphere falling on the trampolines (maintainer, 2026-10-20): at 2 mm the sphere's "
    "impact on the heterogeneous circle trampoline exploded.",
    {"Contact.CDType.VertexSdf.band": 5e-3},
)
TRAMPOLINE = [TRAMPOLINE_DETECTION, SPHERE_BAND]

SCENES = {
    # The six scene configs that predate the explicit form: their values are kept, plus the 2 mm band.
    "basket_of_fruit": Scene(
        "basket_of_fruit.Params_3_10_15",
        "final_scenes/basket_of_fruit.h5 (Young's modulus 1e5 to 1e8)",
        [_stiff_budget("3 substeps of 10 x 15 = 450 sweeps"), *SOLVERS, *CONTACT],
    ),
    "celebration": Scene(
        "celebration.Params_1_5_30",
        "final_scenes/celebration.h5 (Young's modulus 1e6 to 1e8)",
        [
            _stiff_budget("1 x 5 x 30 = 150 sweeps"),
            *SOLVERS,
            *CONTACT,
            Rule(
                "max_dist 30 cm, scaled to the geometry (maintainer): the base plate is 10 cm thick and the "
                "letters' edges ~4 cm.",
                {"Contact.CDType.VertexSdf.max_dist": 0.3},
            ),
        ],
        ["The previous scene config's 1 cm band (for the 10 cm base plate) is now 2 mm, like every scene's."],
    ),
    "mat_on_stilts": Scene(
        "mat_on_stilts.Params.5Substeps",
        "final_scenes/mat_on_stilts_3.h5 (Young's modulus 1e6 to 1e8)",
        [
            _stiff_budget(
                "5 substeps of 5 x 5 = 125 sweeps; with them, EVD and gammaup/gammaupS 0.9/0.85, AAAVBD blows up "
                "once the mat lands on the stilts. 10 x 20 and 10 x 40 were also stable"
            ),
            *SOLVERS,
            *CONTACT,
        ],
        ["The paper's 10-substep file (mat_on_stilts.Params.10Substeps) differs from the 5-substep one only in "
         "substeps, which this config overrides."],
    ),
    "trampoline_soft-stiff": Scene(
        "trampoline_soft-stiff.Params.5Substeps",
        "final_scenes/trampolines/trampoline_{circle,square}_soft-stiff[__with_sphere].h5 (Young's modulus 1e6 to 1e8)",
        [_stiff_budget("5 substeps of 1 x 100 sweeps"), *SOLVERS, *CONTACT, *TRAMPOLINE],
    ),
    "trampoline_stiff-soft": Scene(
        "trampoline_stiff-soft.Params.5Substeps",
        "final_scenes/trampolines/trampoline_{circle,square}_stiff-soft[__with_sphere].h5 (Young's modulus 1e6 to 1e8)",
        [_stiff_budget("5 substeps of 1 x 25 AAAVBD sweeps, 1 x 100 for VBD"), *SOLVERS, *CONTACT,
         *TRAMPOLINE],
    ),
    "trampoline_soft-soft": Scene(
        "trampoline_soft-stiff.Params.5Substeps",
        "final_scenes/trampolines/trampoline_{circle,square}_soft-soft[__with_sphere].h5 (homogeneous, Young's "
        "modulus 1e6)",
        [
            _budget(20, "Homogeneous and soft, about 100 sweeps per step by the maintainer's guidance",
                    "5 substeps of 1 x 100 sweeps, for the soft-stiff trampolines"),
            *SOLVERS,
            *CONTACT,
            *TRAMPOLINE,
        ],
        ["The paper has no parameter file for the soft-soft trampolines; the source is the soft-stiff one's."],
    ),
    # The other paper configs: the durable rules only, budgets unchanged
    "trampoline_stiff_sphere": Scene(
        "trampoline_stiff_sphere.Params.5Substeps",
        "the trampolines with a sphere (final_scenes/trampolines/*__with_sphere.h5; which variant is not recorded)",
        [*SOLVERS, *CONTACT, *TRAMPOLINE],
        ["Budgets kept: 5 substeps of 1 x 100 AAAVBD sweeps (1 x 400 for VBD), one contact detection pass per "
         "substep. The other trampoline scene configs use 1 substep of 5 x 80 instead."],
    ),
    "capsules_into_bowl": Scene(
        "capsules_into_bowl",
        "final_scenes/capsule_bowl.h5 (Young's modulus 1e5 to 1e6)",
        [*SOLVERS, *CONTACT],
        ["Budgets kept: 5 substeps of 3 x 3 = 45 sweeps per frame, below the guidance's ~100 for soft higher-res "
         "meshes (230k tets), though not a stiff scene."],
    ),
    "empty_basket": Scene(
        "empty_basket.Params_3_10_5",
        "probably final_scenes/short_basket.h5 (the woven basket alone, homogeneous, Young's modulus 1e7; not "
        "recorded)",
        [*SOLVERS, *CONTACT],
        ["Budgets kept: 3 substeps of 10 x 5 = 150 sweeps per frame, a little below the guidance's ~200 for "
         "Young's modulus ~1e7."],
    ),
    "basket_5_5_40": Scene(
        "basket_5_5_40",
        "a basket scene, probably final_scenes/basket_of_oranges.h5 (Young's modulus 1e6 to 1e8; not recorded)",
        [*SOLVERS, *CONTACT],
        ["Budgets kept: 5 substeps of 5 x 40 = 1000 sweeps per frame."],
    ),
    # Contact-free scenes (VertexSdf.deactivate = 1): the contact rules don't apply, Newton keeps its budget
    "birthday_banner": Scene(
        "birthday_banner.Params_1_2000",
        "the birthday banner (scene file not in final_scenes; tetrahedra/banner_regions.mesh)",
        [*SOLVERS],
        ["Contact detection is off in the source (VertexSdf.deactivate = 1), so its contact parameters are kept.",
         "Budgets kept: 1 x 2000 sweeps per step."],
    ),
    **{
        f"beam{y}": Scene(
            f"beam{y}.GroundTruth",
            f"final_scenes/bending_beams/small_beam{'_het' if y == '_het' else 'Y' + y}.h5 ("
            + ("Young's modulus 1e6 to 1e8" if y == "_het" else f"homogeneous, Young's modulus {y}")
            + ")",
            [*SOLVERS],
            ["Contact-free (single body, VertexSdf.deactivate = 1): contact parameters kept.",
             "Budgets kept: the paper's ground-truth budgets, chosen to match Newton's behaviour "
             "(accelerated-vbd-extras/parameters/gpu/beam.GroundTruth.Notes.txt)."]
            + (["The paper's beam1e7.GroundTruth.5Substeps (5 substeps of 20 AAAVBD / 500 VBD sweeps) is a variant "
                "of this scene."] if y == "1e7" else []),
        )
        for y in ("1e5", "1e6", "1e7", "1e8", "_het")
    },
}


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
    source = _flatten(config.read(CONFIGS / "paper" / f"{scene.source}.toml"))
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
        f"Source: configs/paper/{scene.source}.toml. Keys not listed below have the source's values.",
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
