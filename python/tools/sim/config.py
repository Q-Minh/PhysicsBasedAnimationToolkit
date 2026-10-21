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
        "replaced by Solver.Convergence.astol, the absolute stationarity tolerance (m): the gradient test now "
        "passes below max(rstol * RMS(x~ - x^t), astol), so the former floor f is astol = rstol * f "
        "(1e-8 m for the former defaults; astol's default is 1e-6 m)"
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
    args = parser.parse_args()
    if args.command == "convert":
        text = dumps(read(args.input), [f"Converted from {args.input}"])
        pathlib.Path(args.output).write_text(text, encoding="utf-8")
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
