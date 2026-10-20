"""TEMPORARY: convert the paper's GPU parameter files to TOML configs, and check the conversion.

This module exists only until the paper's parameter files are finalized; then remove it. Its reusable parts
are not paper-specific and may be kept by turning them into a general tool instead: converting HDF5
parameter files to TOML configs, regenerating ``configs/defaults.toml``, and checking that the loaders read a
config exactly as its source.

Usage, from the repository root:

    python -m python.tools.sim.paper_configs make --extras path/to/accelerated-vbd-extras
    python -m python.tools.sim.paper_configs check --extras path/to/accelerated-vbd-extras

``make`` converts the extras repository's **committed** ``parameters/gpu/*.h5`` files (uncommitted local
changes are ignored, by the maintainer's choice) to ``configs/paper/<stem>.toml``, following the rules in
``configs/README.md`` ("Paper configs"), and regenerates ``configs/defaults.toml``. Run it after a change to
the serialized parameters (a parameter added, removed or given a new default).

``check`` verifies that each paper ``.toml`` loads to exactly the same parameters as its ``.h5`` (both through
the loaders), and that every attribute of the raw ``.h5`` matches the loaded state, except the keys the code
no longer serializes. It exits nonzero on any difference or ConfigWarning.

Both commands build a SimulationState from ``--problem`` (default: the beam Dump in the git-ignored ``data/``).
"""

import argparse
import datetime
import hashlib
import pathlib
import subprocess
import sys
import tempfile
import warnings

import h5py
import numpy as np
import warp as wp

from . import config
from . import state as st
from .gpu.vbd.params import STENCIL_GRADIENT_ACCELERATION_KEYS

CONFIGS = pathlib.Path(__file__).parent / "configs"
DEFAULT_PROBLEM = "data/scenarios/MediumLongBendingBeam.Dump.h5:fem"


def _git(extras: pathlib.Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(extras), *args], capture_output=True, text=True, check=True).stdout.strip()


def _committed_sources(extras: pathlib.Path, tmp: pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    """(repo-relative path, temporary copy) of each committed parameters/gpu/*.h5 of the extras repository."""
    names = _git(extras, "ls-tree", "--name-only", "HEAD", "parameters/gpu/").splitlines()
    out = []
    for rel in sorted(n for n in names if n.endswith(".h5")):
        blob = subprocess.run(["git", "-C", str(extras), "show", f"HEAD:{rel}"], capture_output=True, check=True).stdout
        copy = tmp / pathlib.PurePosixPath(rel).name
        copy.write_bytes(blob)
        out.append((rel, copy))
    return out


def _make_state(problem: str) -> st.SimulationState:
    fem_cpu, kinematics, geometry = st.load_problem(problem)
    return st.SimulationState(fem_cpu, st.load_params(fem_cpu), kinematics, geometry)


def _flatten(d: dict, prefix: str = ""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flatten(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}"


def make(extras: pathlib.Path, problem: str) -> None:
    defaults = config.params_dict(_make_state(problem))
    commit = _git(extras, "rev-parse", "--short", "HEAD")
    today = datetime.date.today().isoformat()
    (CONFIGS / "paper").mkdir(parents=True, exist_ok=True)
    (CONFIGS / "defaults.toml").write_text(
        config.dumps(defaults, [
            "Every parameter the GPU simulators serialize, with its code default.",
            f"Generated {today} with config.params_dict on a default SimulationState. Regenerate it when",
            "parameters are added (see README.md). Useful as a reference or as an `extends` base.",
        ]),
        encoding="utf-8",
    )
    with tempfile.TemporaryDirectory() as tmp:
        for rel, blob in _committed_sources(extras, pathlib.Path(tmp)):
            sha = hashlib.sha256(blob.read_bytes()).hexdigest()
            source = config.read(blob)
            solver = source.setdefault("Solver", {})
            augmented = []
            if "Chebyshev" not in solver:
                solver["Chebyshev"] = dict(solver["VBD"], rho=defaults["Solver"]["Chebyshev"]["rho"])
                augmented.append("Solver.Chebyshev: Solver.VBD's values, plus the default rho")
            if "AndersonSolver" not in solver:
                aa = defaults["Solver"]["AndersonSolver"]
                solver["AndersonSolver"] = dict(solver["VBD"], **{k: aa[k] for k in ("m", "beta", "cod_numerical_zero")})
                augmented.append(
                    "Solver.AndersonSolver: Solver.VBD's values, plus the default m, beta, cod_numerical_zero"
                )
            unknown, _ = config.check(source, defaults)
            known = config._known(source, defaults)
            cfg = config.merge(defaults, known)
            from_defaults = sorted(set(_flatten(defaults)) - set(_flatten(known)))
            header = [
                f"Paper parameters, converted from accelerated-vbd-extras/{rel}",
                f"  (extras commit {commit}, sha256 {sha[:16]}), {today}.",
                "Hand-tuned values from the source are unchanged. Edit a copy, not this file (see README.md).",
            ]
            if augmented:
                header += ["", "Added for the accelerated solvers (not hand-tuned):"] + [f"  {a}" for a in augmented]
            if from_defaults:
                # The augmented tables' keys come from Solver.VBD, except the GPU-only Anderson keys
                header += ["", "Not in the source, code defaults:"] + [
                    f"  {k}"
                    for k in from_defaults
                    if not k.startswith(("Solver.Chebyshev.", "Solver.AndersonSolver."))
                    or k.split(".")[-1] in ("linear_solver", "vbd_iters_per_anderson_iter")
                ]
            if unknown:
                header += ["", "Dropped (not serialized by the current code):"] + [f"  {k}" for k in unknown]
            out = CONFIGS / "paper" / f"{pathlib.PurePosixPath(rel).stem}.toml"
            out.write_text(config.dumps(cfg, header), encoding="utf-8")
            print(f"{out.name}: augmented {len(augmented)}, defaults {len(from_defaults)}, dropped {len(unknown)}")


def _serialized(state: st.SimulationState) -> dict:
    """Every serialized parameter of ``state`` as ``{"Group/Sub/attr": value}``."""
    out = {}
    with config._memory_h5() as f:
        st.serialize_params(state, f)
        f.visititems(lambda name, obj: out.update({f"{name}/{k}": np.asarray(v).item() for k, v in obj.attrs.items()}))
    return out


def check(extras: pathlib.Path, problem: str) -> int:
    """Number of paper configs that differ from their sources (0 when all pass)."""
    from_h5, from_toml = _make_state(problem), _make_state(problem)
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for rel, blob in _committed_sources(extras, pathlib.Path(tmp)):
            stem = pathlib.PurePosixPath(rel).stem
            with h5py.File(blob, "r") as f:
                st.deserialize_params(from_h5, f)
                present = {g for g in ("Solver/Chebyshev", "Solver/AndersonSolver") if g in f}
                raw = {}
                f.visititems(lambda n, o: raw.update({f"{n}/{k}": np.asarray(v).item() for k, v in o.attrs.items()}))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                config.load_sim_params(from_toml, [str(CONFIGS / "paper" / f"{stem}.toml")])
            a, b = _serialized(from_h5), _serialized(from_toml)
            # .toml vs .h5 through the loaders; the augmented tables have no .h5 counterpart
            skip = {"Solver/Chebyshev", "Solver/AndersonSolver"} - present
            problems = [f"loaded differently: {k}" for k in a if k.rsplit("/", 1)[0] not in skip and a[k] != b[k]]
            # raw .h5 attributes vs the loaded state, except keys the code no longer serializes
            for key, value in raw.items():
                group, attr = key.rsplit("/", 1)
                dropped = (attr in STENCIL_GRADIENT_ACCELERATION_KEYS and not group.endswith("AAAVBD")) or (
                    key == "Contact/CDType/VertexSdf/dmin"
                )
                if key not in b:
                    if not dropped:
                        problems.append(f"not loaded: {key}")
                elif float(value) != float(b[key]):
                    problems.append(f"{key}: file {value!r} vs loaded {b[key]!r}")
            problems += [f"warning: {w.message}" for w in caught]
            bad += bool(problems)
            print(f"{stem}: {'OK' if not problems else 'MISMATCH'}" + "".join(f"\n  {p}" for p in problems))
    print(f"{bad} mismatching config(s)")
    return bad


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["make", "check"])
    parser.add_argument("--extras", required=True, type=pathlib.Path, help="The accelerated-vbd-extras repository.")
    parser.add_argument("--problem", default=DEFAULT_PROBLEM, help="file.h5:group of a FemElastoDynamics to build states from.")
    args = parser.parse_args()
    wp.init()
    if args.command == "make":
        make(args.extras, args.problem)
    else:
        sys.exit(check(args.extras, args.problem))


if __name__ == "__main__":
    main()
