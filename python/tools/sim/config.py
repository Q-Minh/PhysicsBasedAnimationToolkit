"""Layered TOML configs for the GPU simulators' parameters. ``configs/README.md`` is the user guide.

A config mirrors the parameter layout that :func:`state.serialize_params` writes to HDF5: each group
is a TOML table (``Solver/VBD`` is ``[Solver.VBD]``) and each attribute a key. Enums are their integer
values and bools are 0/1, as in the HDF5 files. Floats are written exactly (``repr``), so the two
formats convert losslessly: a value saved from a C++ float reads e.g. ``0.0010000000474974513``, and
shortening it to ``0.001`` would change parameters that the host uses in double precision (``dt``).

Loading (:func:`load_sim_params`) merges sources left to right, later values winning:

1. each ``--sim-params`` source, a ``.toml`` config or an HDF5 parameter file (``file.h5[:group]``).
   A TOML file may start with ``extends = ["base.toml", ...]`` (paths relative to it), which are
   merged first, in order;
2. each ``--set Table.Sub.key=value`` override, ``value`` being a TOML literal.

The merged config is then checked against the keys the current code serializes. Unknown keys are
ignored and missing keys keep their current values (the code defaults); both are reported as
:class:`ConfigWarning`. Keys that a newer parameter replaced (:data:`DEPRECATED`) are ignored too,
with a warning that says what replaced them, and renamed keys (:data:`RENAMED`) are read under their
new names, with a warning.

Command line::

    python -m python.tools.sim.config convert params.h5 params.toml   # HDF5 parameter file -> TOML
    python -m python.tools.sim.config show a.toml [b.toml ...] [--set Solver.VBD.gtol=1e-4]
    python -m python.tools.sim.config params [Solver.Newton]   # every settable key: default, type, enum names, docs

``params`` lists the keys of ``configs/defaults.toml`` (every key the code serializes) with their default,
type, enum values by name, and the description the code gives them (``DocField`` docs, the C++ bindings'
property docs), "(no description in the code)" otherwise.
"""

import argparse
import copy
import pathlib
import tomllib
import uuid
import warnings
from typing import TYPE_CHECKING, Iterable

import h5py
import numpy as np
import tomli_w

if TYPE_CHECKING:
    from .state import SimulationState

EXTENDS = "extends"

# Keys replaced by newer parameters: ignored when loading, with a ConfigWarning naming the replacement
DEPRECATED = {
    "Contact.CDType.VertexSdf.dmin": (
        "replaced by Contact.CDType.VertexSdf.band: contacts are now detected below Contact.ContactDynamics.dmin "
        "+ band, so that resting contacts stay detected and their multipliers persist. The old absolute "
        "threshold (-1e-4 detected penetrating vertices only) is ignored, and band keeps its value"
    ),
    "Solver.Convergence.displacement_floor": (
        "replaced by Solver.Convergence.astol, the absolute stationarity tolerance as a fraction of the scene's "
        "external acceleration |a|: the gradient test now passes below max(rstol * RMS(x~ - x^t), astol * |a| * "
        "h^2), so the former floor f corresponds to astol = rstol * f / (|a| h^2)"
    ),
}

# Keys renamed without a change of meaning: their values move to the new key, with a ConfigWarning
RENAMED = {
    "Solver.Convergence.rtol": "Solver.Convergence.rstol",
}


class ConfigWarning(UserWarning):
    """A config has unknown keys, or lacks keys that keep their current values."""


def h5_to_dict(grp: h5py.Group) -> dict:
    """Convert an HDF5 parameter group to a nested dict: subgroups become tables, attributes keys."""
    out = {k: _to_toml_value(v) for k, v in grp.attrs.items()}
    for k, sub in grp.items():
        if isinstance(sub, h5py.Group):
            out[k] = h5_to_dict(sub)
    return out


def _to_toml_value(v):
    if isinstance(v, np.ndarray):
        return [_to_toml_value(e) for e in v.tolist()]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, bytes):
        v = v.decode()
    return v


def dict_to_h5(d: dict, grp: h5py.Group) -> None:
    """Write a nested config dict into an HDF5 group, the inverse of :func:`h5_to_dict`."""
    for k, v in d.items():
        if isinstance(v, dict):
            dict_to_h5(v, grp.require_group(k))
        else:
            grp.attrs[k] = int(v) if isinstance(v, bool) else v


def merge(base: dict, over: dict) -> dict:
    """Recursively merge ``over`` into a copy of ``base``; values of ``over`` win."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def read(source: str | pathlib.Path) -> dict:
    """Read one source: a ``.toml`` config (resolving ``extends``) or an HDF5 file ``file.h5[:group]``."""
    return _read(str(source), [])


def _read(source: str, stack: list[pathlib.Path]) -> dict:
    if source.lower().endswith(".toml"):
        path = pathlib.Path(source).resolve()
        if path in stack:
            raise ValueError(f"Config extends itself: {' -> '.join(map(str, stack + [path]))}")
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
        parents = cfg.pop(EXTENDS, [])
        if isinstance(parents, str):
            parents = [parents]
        out = {}
        for parent in parents:
            out = merge(out, _read(str(path.parent / parent), stack + [path]))
        return merge(out, cfg)
    # file.h5[:group], where a single letter before the only ':' is a Windows drive
    file_path, sep, group_path = source.rpartition(":")
    if not sep or len(file_path) <= 1:
        file_path, group_path = source, ""
    with h5py.File(file_path, "r") as f:
        return h5_to_dict(f[group_path] if group_path else f)


def parse_override(spec: str) -> dict:
    """Parse ``Table.Sub.key=value`` (``value`` a TOML literal) into a nested dict."""
    key, sep, value = spec.partition("=")
    if not sep or not key.strip():
        raise ValueError(f"Override must be Table.key=value, got '{spec}'")
    try:
        parsed = tomllib.loads(f"v = {value.strip()}")["v"]
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Override value of '{key}' is not a TOML literal: {value}") from e
    out = parsed
    for part in reversed(key.strip().split(".")):
        out = {part: out}
    return out


def resolve(sources: Iterable[str], overrides: Iterable[str] = ()) -> dict:
    """Merge the sources, then the overrides, left to right."""
    cfg = {}
    for source in sources:
        cfg = merge(cfg, read(source))
    for spec in overrides:
        cfg = merge(cfg, parse_override(spec))
    return cfg


def _memory_h5() -> h5py.File:
    return h5py.File(f"{uuid.uuid4().hex}.h5", "w", driver="core", backing_store=False)


def params_dict(state: "SimulationState") -> dict:
    """The state's current parameters as a config dict (every key the code serializes)."""
    from .state import serialize_params

    with _memory_h5() as f:
        serialize_params(state, f)
        return h5_to_dict(f)


def check(cfg: dict, reference: dict) -> tuple[list[str], list[str]]:
    """Compare a config with a reference config of every known key.

    Returns:
        (unknown, missing) dotted key paths. A table that is missing altogether is listed once.
    """
    unknown, missing = [], []

    def walk(c: dict, r: dict, prefix: str):
        for k, v in c.items():
            path = f"{prefix}{k}"
            if k not in r or isinstance(v, dict) != isinstance(r[k], dict):
                unknown.append(path)
            elif isinstance(v, dict):
                walk(v, r[k], f"{path}.")
        for k, v in r.items():
            if k not in c or isinstance(c[k], dict) != isinstance(v, dict):
                missing.append(f"{prefix}{k}" + (" (whole table)" if isinstance(v, dict) else ""))

    walk(cfg, reference, "")
    return unknown, missing


def _rename(cfg: dict, source: str) -> dict:
    """A copy of ``cfg`` with the :data:`RENAMED` keys moved to their new names (an explicit new key wins),
    warning about each."""
    cfg = copy.deepcopy(cfg)
    for old, new in RENAMED.items():
        *old_tables, old_key = old.split(".")
        table = cfg
        for t in old_tables:
            table = table.get(t) if isinstance(table, dict) else None
        if not isinstance(table, dict) or old_key not in table:
            continue
        value = table.pop(old_key)
        *new_tables, new_key = new.split(".")
        target = cfg
        for t in new_tables:
            target = target.setdefault(t, {})
        target.setdefault(new_key, value)
        warnings.warn(f"{source}: {old} is renamed {new}; using its value", ConfigWarning, stacklevel=3)
    return cfg


def _known(cfg: dict, reference: dict) -> dict:
    return {
        k: _known(v, reference[k]) if isinstance(v, dict) else v
        for k, v in cfg.items()
        if k in reference and isinstance(v, dict) == isinstance(reference[k], dict)
    }


def apply(state: "SimulationState", cfg: dict, source: str = "sim-params") -> None:
    """Load a config into the state's parameters, warning about unknown and missing keys.

    Call ``state.reset()`` afterwards, as after any parameter change.
    """
    from .state import deserialize_params

    reference = params_dict(state)
    cfg = _rename(cfg, source)
    unknown, missing = check(cfg, reference)
    for key in [k for k in unknown if k in DEPRECATED]:
        unknown.remove(key)
        warnings.warn(f"{source}: {key} is deprecated, ignored: {DEPRECATED[key]}", ConfigWarning, stacklevel=2)
    if unknown:
        warnings.warn(
            f"{source}: unknown keys, ignored: {', '.join(unknown)}", ConfigWarning, stacklevel=2
        )
    if missing:
        warnings.warn(
            f"{source}: keys not set, keeping their current values: {', '.join(missing)}",
            ConfigWarning,
            stacklevel=2,
        )
    with _memory_h5() as f:
        dict_to_h5(_known(cfg, reference), f)
        deserialize_params(state, f)


def load_sim_params(
    state: "SimulationState", sources: Iterable[str], overrides: Iterable[str] = ()
) -> dict:
    """Resolve the sources and overrides (see the module docstring) and apply them to the state."""
    sources, overrides = list(sources), list(overrides)
    cfg = resolve(sources, overrides)
    apply(state, cfg, source=" + ".join(sources + [f"--set {o}" for o in overrides]))
    return cfg


DEFAULTS_TOML = pathlib.Path(__file__).parent / "configs" / "defaults.toml"
# Enums of keys whose parameter objects don't hold them as enums
_ENUM_KEYS = {"Integration.init_strategy": ("pbatoolkit", "pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization")}
# Keys the GPU solvers ignore (checked in gpu/: vbd/params.py's Params copies neither k, kp nor e_penalty_stiffness,
# and no kernel reads its betaR; gpu/newton/solver.py's Params never reads linear_solver)
_VBD_FAMILY = ("VBD", "AAAVBD", "Chebyshev", "AndersonSolver")
GPU_IGNORED = {
    "Solver.Newton.linear_solver": "GPU: ignored; always Warp's Jacobi-preconditioned CG (Solver.Newton.LinearSolver)",
    **{f"Solver.{s}.e_penalty_stiffness": "GPU: ignored; the GPU's contact penalty is "
       "Contact.ContactDynamics.penalty_adaptivity" for s in _VBD_FAMILY},
    **{f"Solver.{s}.betaR": "GPU: ignored; copied to the GPU params but no kernel reads it" for s in _VBD_FAMILY},
    **{f"Solver.{s}.{k}": "GPU: ignored; a C++ runtime counter, not copied to the GPU" for s in _VBD_FAMILY for k in ("k", "kp")},
}
# SimulationState attributes that serialize_params writes directly, with no parameter object (or docs) behind them
_STATE_KEYS = {"Integration.dt", "Integration.substeps", "Integration.bdf_scheme", "Integration.init_strategy"}


def _param_sources() -> dict:
    """Table path -> the parameter objects (or DocField classes) its keys come from, as serialize_params
    writes them; looked up in order. CPU objects only."""
    from pbatoolkit import pbat

    from . import gpu
    from .gpu.contact.mesh import ogc, pairs, sd

    vbd = pbat.sim.algorithm.vbd.Params()
    newton = pbat.sim.algorithm.newton.Params()
    return {
        "Integration": [],
        "Solver.VBD": [vbd],
        "Solver.AAAVBD": [gpu.vbd.params.AaaVbdCpuParams, vbd],
        "Solver.Chebyshev": [pbat.sim.algorithm.vbd.ChebyshevParams(), vbd],
        "Solver.AndersonSolver": [gpu.vbd.params.AndersonCpuParams(pbat.sim.algorithm.vbd.Params()), vbd],
        "Solver.Newton": [gpu.newton.solver.NewtonGpuParams, newton],
        "Solver.Newton.newton": [newton.newton],
        # A default Params has no line search; serialize_params writes the one SimulationState sets
        "Solver.Newton.newton.line_search": [pbat.math.optimization.BackTrackingLineSearch()],
        "Solver.Newton.LinearSolver": [gpu.newton.solver.LinearSolverParams],
        "Solver.Convergence": [gpu.vbd.convergence.ConvergenceParams],
        "Contact.CDType.OGC": [ogc.OgcParams],
        "Contact.CDType.VertexSdf": [sd.Params],
        "Contact.ContactStorage": [pairs.Params],
        "Contact.ContactDynamics": [gpu.contact.dynamics.Params],
    }


def _describe(sources: list, name: str) -> tuple[bool, type | None, str]:
    """(found, enum type, description) of key ``name`` from the first source that has it."""
    import enum

    from .common.fields import DocField

    for source in sources:
        cls = source if isinstance(source, type) else type(source)
        attr = getattr(cls, name, None)
        if isinstance(attr, DocField):
            default = attr.default
            return True, type(default) if isinstance(default, enum.Enum) else None, (attr.__doc__ or "").strip()
        if attr is not None and not isinstance(source, type):
            value = getattr(source, name)
            return True, type(value) if isinstance(value, enum.Enum) else None, (attr.__doc__ or "").strip()
    return False, None, ""


def flatten(cfg: dict, prefix: str = ""):
    for k, v in cfg.items():
        if isinstance(v, dict):
            yield from flatten(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def describe_params(prefix: str = "") -> list[dict]:
    """Every key a config can set (``configs/defaults.toml``) under ``prefix``: ``key`` (dotted TOML path),
    ``default``, ``type`` (int, float, bool or enum), ``enum`` ({value: name} or None), ``description`` ("" when
    the code has none), ``gpu`` (a note when the GPU solvers ignore the key, ``GPU_IGNORED``, else "") and
    ``found`` (whether a parameter object holds the key)."""
    import enum
    import importlib

    sources = _param_sources()
    rows = []
    for key, default in flatten(read(DEFAULTS_TOML)):
        if prefix and not (key == prefix or key.startswith(prefix.rstrip(".") + ".")):
            continue
        table, _, name = key.rpartition(".")
        found, enum_type, description = _describe(sources.get(table, []), name)
        found = found or key in _STATE_KEYS
        if key in _ENUM_KEYS:
            module, path = _ENUM_KEYS[key]
            enum_type = importlib.import_module(module)
            for part in path.split("."):
                enum_type = getattr(enum_type, part)
            found = True
        names = None
        if enum_type is not None and issubclass(enum_type, enum.Enum):
            names = {int(m.value): n for n, m in enum_type.__members__.items()}
        kind = "enum" if names else "float" if isinstance(default, float) else "int"
        if not names and isinstance(default, int):
            # Bools serialize as 0/1: a DocField or binding default tells them apart
            for source in sources.get(table, []):
                value = getattr(source, name, None)
                value = getattr(value, "default", value)
                if isinstance(value, bool):
                    kind = "bool (0/1)"
                    break
        rows.append({"key": key, "default": default, "type": kind, "enum": names, "description": description,
                     "gpu": GPU_IGNORED.get(key, ""), "found": found})
    return rows


def format_params(rows: list[dict]) -> str:
    """One line per key: ``key = default  (type[: enum values])  description``."""
    lines = []
    for r in rows:
        kind = r["type"]
        if r["enum"]:
            kind += ": " + ", ".join(f"{v} {n}" for v, n in sorted(r["enum"].items()))
        gpu = f" ({r['gpu']})" if r.get("gpu") else ""
        lines.append(f"{r['key']} = {r['default']!r}  ({kind})  {r['description'] or '(no description in the code)'}{gpu}")
    return "\n".join(lines)


def dumps(cfg: dict, header: Iterable[str] = ()) -> str:
    """Format a config as TOML, preceded by ``header`` lines as comments."""
    lines = "".join(f"# {line}\n" if line else "#\n" for line in header)
    return lines + ("\n" if lines else "") + tomli_w.dumps(cfg)


def main():
    parser = argparse.ArgumentParser(description="Simulation-parameter configs (see configs/README.md)")
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert", help="Convert an HDF5 parameter file to TOML")
    convert.add_argument("input", help="file.h5[:group], e.g. a GUI export or a simulate.py output's file.h5:params")
    convert.add_argument("output", help="Output .toml file")
    show = sub.add_parser("show", help="Print the merged config of the sources and overrides")
    show.add_argument("sources", nargs="+", help=".toml configs or HDF5 parameter files, merged left to right")
    show.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE")
    params = sub.add_parser("params", help="List every settable key: default, type, enum values, description")
    params.add_argument("prefix", nargs="?", default="", help="Only keys under this table path, e.g. Solver.Newton")
    args = parser.parse_args()
    if args.command == "convert":
        text = dumps(read(args.input), [f"Converted from {args.input}"])
        pathlib.Path(args.output).write_text(text, encoding="utf-8")
    elif args.command == "params":
        rows = describe_params(args.prefix)
        if not rows:
            raise SystemExit(f"No keys under {args.prefix!r}; tables: {', '.join(sorted(_param_sources()))}")
        print(format_params(rows))
    else:
        print(dumps(resolve(args.sources, args.overrides)), end="")


import tempfile
import unittest


class TestConfig(unittest.TestCase):
    def test_h5_round_trip_is_exact(self):
        gtol = float(np.float32(1e-3))  # as pbat's float params arrive, 0.0010000000474974513
        with _memory_h5() as f:
            g = f.create_group("Solver/VBD")
            g.attrs["gtol"] = gtol
            g.attrs["n_max_iters"] = np.int32(10)
            cfg = tomllib.loads(dumps(h5_to_dict(f), ["header"]))
        self.assertEqual(cfg, {"Solver": {"VBD": {"gtol": gtol, "n_max_iters": 10}}})
        with _memory_h5() as f:
            dict_to_h5(cfg, f)
            self.assertEqual(f["Solver/VBD"].attrs["gtol"], gtol)
            self.assertEqual(f["Solver/VBD"].attrs["n_max_iters"], 10)

    def test_extends_and_overrides_merge_left_to_right(self):
        with tempfile.TemporaryDirectory() as d:
            base = pathlib.Path(d, "base.toml")
            base.write_text("[Solver.VBD]\ngtol = 0.1\nn_max_iters = 3\n")
            sub = pathlib.Path(d, "sub", "child.toml")
            sub.parent.mkdir()
            sub.write_text('extends = ["../base.toml"]\n[Solver.VBD]\ngtol = 0.2\n')
            cfg = resolve([str(sub)], ["Solver.VBD.n_max_iters=7", "Integration.dt=1e-3"])
            self.assertEqual(cfg, {"Solver": {"VBD": {"gtol": 0.2, "n_max_iters": 7}}, "Integration": {"dt": 1e-3}})
            base.write_text('extends = "sub/child.toml"\n')
            with self.assertRaises(ValueError):
                read(sub)

    def test_check_reports_unknown_and_missing_keys(self):
        reference = {"Solver": {"VBD": {"gtol": 1.0, "k": 0}, "Newton": {"n": 1}}, "Integration": {"dt": 0.01}}
        cfg = {"Solver": {"VBD": {"gtol": 2.0, "typo": 1}}, "Integration": {"dt": {"x": 1}}}
        unknown, missing = check(cfg, reference)
        self.assertEqual(unknown, ["Solver.VBD.typo", "Integration.dt"])
        self.assertEqual(missing, ["Solver.VBD.k", "Solver.Newton (whole table)", "Integration.dt"])
        self.assertEqual(_known(cfg, reference), {"Solver": {"VBD": {"gtol": 2.0}}, "Integration": {}})

    def test_describe_params(self):
        rows = {r["key"]: r for r in describe_params()}
        # Every serialized key maps to a parameter object, so the listing follows the serializers
        self.assertEqual([k for k, r in rows.items() if not r["found"]], [])
        self.assertEqual(set(rows), {k for k, _ in flatten(read(DEFAULTS_TOML))})
        vlinsolve = rows["Solver.AAAVBD.vlinsolve"]
        self.assertEqual((vlinsolve["type"], vlinsolve["enum"][0]), ("enum", "Inverse"))
        self.assertIn("Relative", rows["Solver.Newton.LinearSolver.rtol"]["description"])
        self.assertEqual(rows["Solver.Newton.spd_correction"]["type"], "enum")
        self.assertEqual(rows["Integration.init_strategy"]["type"], "enum")
        self.assertEqual(rows["Contact.CDType.VertexSdf.deactivate"]["type"], "bool (0/1)")
        newton = describe_params("Solver.Newton")
        self.assertTrue(newton and all(r["key"].startswith("Solver.Newton.") for r in newton))
        self.assertEqual(describe_params("Solver.Newt"), [])  # whole table names only
        self.assertIn("(no description in the code)", format_params([rows["Integration.dt"]]))
        # GPU notes name keys that exist
        self.assertLessEqual(set(GPU_IGNORED), set(rows))
        self.assertIn("(GPU: ignored; always Warp's Jacobi-preconditioned CG",
                      format_params([rows["Solver.Newton.linear_solver"]]))
        self.assertEqual(rows["Solver.Newton.LinearSolver.rtol"]["gpu"], "")

    def test_renamed_keys_move_to_their_new_names(self):
        cfg = {"Solver": {"Convergence": {"rtol": 0.05, "ftol": 0.5}}}
        with self.assertWarns(ConfigWarning):
            renamed = _rename(cfg, "test")
        self.assertEqual(renamed, {"Solver": {"Convergence": {"rstol": 0.05, "ftol": 0.5}}})
        self.assertIn("rtol", cfg["Solver"]["Convergence"])  # the input is left as is
        # An explicit new key wins over the renamed one
        with self.assertWarns(ConfigWarning):
            both = _rename({"Solver": {"Convergence": {"rtol": 0.05, "rstol": 0.2}}}, "test")
        self.assertEqual(both["Solver"]["Convergence"], {"rstol": 0.2})

    def test_override_values_are_toml_literals(self):
        self.assertEqual(parse_override("A.b = true"), {"A": {"b": True}})
        with self.assertRaises(ValueError):
            parse_override("A.b=not a literal")


if __name__ == "__main__":
    main()
