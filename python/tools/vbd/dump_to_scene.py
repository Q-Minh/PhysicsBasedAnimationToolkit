# type: ignore
"""
Convert an editor Dump (file.h5 with a `fem` group) back into a scene file that the editor, the CPU offline
simulator and the GPU simulators' `--scene` read.

The Dump stores the assembled FEM problem, so the scene's per-element materials and per-body loads are
recovered from it:
- each connected component becomes a body;
- Young's modulus and Poisson's ratio come from the Lamé parameters (one quadrature point per element);
- the density from the lumped masses, which must be uniform per body;
- the external acceleration from fext / m, which must be uniform per body (bext is zero);
- the initial velocities are the Dump's v;
- recovered material and load values are rounded to 5 significant digits (float32 assembly leaves
  relative offsets of a few 1e-6), so that authored inputs like 1e6, 0.45 or -9.81 come back exactly
  and the rebuilt FEM equals the Dump's;
- the transforms come from the Dump's `kinematics` group, or, without one, a FIXED transform on the nodes
  with nonzero dmask, active forever.
The conversion is checked by rebuilding the FEM from the scene (scenario.make_fem_dynamics_object) and
comparing it with the Dump's. A Dump taken mid-simulation (t0 > 0) is rejected, since a scene starts at rest
positions at t = 0.

Usage (from the repo root):
    python -m python.tools.vbd.dump_to_scene dump.h5 scene.h5
"""

import argparse
import gc
import h5py as h5
import numpy as np
from pbatoolkit import pbat
from .ui.tetrahedral_elastodynamics_body import TetrahedralElastodynamicsBody
from .ui.utils import transform_library as tl
from . import scenario

_KINDS = {0: tl.TransformType.FIXED, 1: tl.TransformType.TRANSLATE, 2: tl.TransformType.L_ROTATE, 3: tl.TransformType.L_TRANSLATE}


def _load_fem(path: str) -> pbat.sim.dynamics.FemElastoDynamics:
    archive = pbat.io.Archive(path, flags=pbat.io.AccessMode.ReadOnly)
    fem = pbat.sim.dynamics.FemElastoDynamics()
    fem.deserialize(archive["fem"])
    archive = None
    gc.collect()
    return fem


def _round(values) -> np.ndarray:
    """Round to 5 significant digits, above float32 assembly's relative error."""
    v = np.asarray(values, dtype=np.float64)
    scale = 10.0 ** (4 - np.floor(np.log10(np.abs(np.where(v == 0, 1, v)))))
    return np.round(v * scale) / scale


def _uniform(values: np.ndarray, what: str, rtol: float = 1e-4) -> float:
    values = np.asarray(values, dtype=np.float64)
    v = float(_round(np.mean(values)))
    if not np.allclose(values, v, rtol=rtol, atol=rtol * abs(v)):
        raise ValueError(f"{what} is not uniform (range [{values.min()}, {values.max()}]); not supported")
    return v


def _transforms(f: h5.File, fem, names: list[str], offsets: np.ndarray) -> list[tl.PrimitiveTransform]:
    """The Dump's constraints as editor transforms, with their nodes per body."""
    if "kinematics" in f:
        g = f["kinematics"]
        if float(g.attrs["t0"]) != 0.0:
            raise ValueError("The Dump was taken mid-simulation (t0 > 0); a scene starts at t = 0")
        node_offsets, nodes = g["node_offsets"][()], g["nodes"][()]
        specs = [
            (
                str(g["name"][k].decode() if isinstance(g["name"][k], bytes) else g["name"][k]),
                int(g["kind"][k]),
                float(g["begin"][k]),
                float(g["duration"][k]),
                g["direction"][k],
                g["point"][k],
                float(g["rate"][k]),
                nodes[node_offsets[k] : node_offsets[k + 1]],
            )
            for k in range(node_offsets.shape[0] - 1)
        ]
    else:
        fixed = np.flatnonzero(np.asarray(fem.dmask).ravel())
        specs = [("Fixed", 0, 0.0, np.inf, np.zeros(3), np.zeros(3), 0.0, fixed)] if fixed.size else []
    transforms = []
    for name, kind, begin, duration, direction, point, rate, knodes in specs:
        ttype = _KINDS[kind]
        if ttype == tl.TransformType.FIXED:
            tr = tl.FixedTransform(name, begin, duration)
        elif ttype == tl.TransformType.TRANSLATE:
            tr = tl.TranslateTransform(name, begin, duration, direction, rate)
        elif ttype == tl.TransformType.L_ROTATE:
            tr = tl.LocalRotateTransform(name, begin, duration, direction, point, np.rad2deg(rate))
        else:
            tr = tl.LocalTranslateTransform(name, begin, duration, point, rate)
        tr.adjust()
        for b, mesh in enumerate(names):
            local = knodes[(knodes >= offsets[b]) & (knodes < offsets[b + 1])] - offsets[b]
            tr._mesh_dirichlet_nodes[mesh] = np.asarray(local, dtype=np.int64)
        transforms.append(tr)
    return transforms


def dump_to_scene(dump_path: str, scene_path: str, name: str | None = None):
    fem = _load_fem(dump_path)
    X, E = fem.X, fem.E
    _, _, XCC, ECC, n_components = pbat.graph.sorted_connected_component_ordering(X, E)
    offsets = np.searchsorted(XCC, np.arange(n_components + 1))
    if not (np.all(np.diff(XCC) >= 0) and np.all(np.diff(ECC) >= 0)):
        raise ValueError("The Dump's FEM is not sorted by connected components")
    if fem.wgU.size != E.shape[1]:
        raise ValueError("Expected 1 elasticity quadrature point per element")
    mu, lam = fem.lamegU[0].astype(np.float64), fem.lamegU[1].astype(np.float64)
    Ye, nue = _round(mu * (3 * lam + 2 * mu) / (lam + mu)), _round(lam / (2 * (lam + mu)))
    # Lumped masses with unit density, from the same quadrature as the editor's
    unit = scenario.make_fem_dynamics_object([_body("probe", X.T, E.T, Ye, nue, np.ones(E.shape[1]), np.zeros(3), X.T * 0)])
    base = name or dump_path.replace("\\", "/").split("/")[-1].removesuffix(".h5").removesuffix(".Dump")
    bodies = []
    for b in range(n_components):
        nb, eb = slice(offsets[b], offsets[b + 1]), np.flatnonzero(ECC == b)
        rho = _uniform(fem.m[nb] / unit.m[nb], f"Body {b} density")
        aext = np.array(
            [
                _uniform(fem.fext[d, nb].astype(np.float64) / fem.m[nb], f"Body {b} external acceleration[{d}]")
                for d in range(3)
            ]
        )
        bodies.append(
            _body(
                f"{base} - {b}",
                X[:, nb].T,
                E[:, eb].T - offsets[b],
                Ye[eb],
                nue[eb],
                np.full(eb.size, rho),
                aext,
                fem.v[:, nb].T,
            )
        )
    names = [body.name for body in bodies]
    with h5.File(dump_path, "r") as f:
        transforms = _transforms(f, fem, names, offsets)
    library = tl.TransformLibrary()
    for mesh in names:
        library.on_mesh_added(mesh)
    for tr in transforms:
        tr.id = len(library.transforms) + 1
        library.transforms.append(tr)
    with h5.File(scene_path, "w") as f:
        grp = f.create_group("fem_tet_elastic_bodies")
        grp.attrs["num_tet_elastic_bodies"] = len(bodies)
        for b, body in enumerate(bodies):
            body.serialize(grp.create_group(f"{b}"))
        f.create_group("static_mesh_colliders").attrs["num_static_mesh_colliders"] = 0
        library.serialize(f.create_group("transform_library"))
        f["recycled_tet_elastic_body_indices"] = np.zeros(0, dtype=int)
    _check(scene_path, fem)


def _body(name, V, T, Ye, nue, rhoe, aext, v0) -> TetrahedralElastodynamicsBody:
    body = TetrahedralElastodynamicsBody()
    body.on_mesh_loaded(
        name, np.asarray(V, dtype=float), np.asarray(T, dtype=np.int64), None, np.asarray(Ye, dtype=float),
        np.asarray(nue, dtype=float), np.asarray(rhoe, dtype=float), np.zeros((T.shape[0], 3)),
        np.asarray(aext, dtype=float), np.asarray(v0, dtype=float), headless=True,
    )
    return body


def _check(scene_path: str, dump_fem):
    """Rebuild the FEM from the written scene and compare it with the Dump's."""
    bodies, meshes, library, names = scenario.read_scene(scene_path)
    fem = scenario.make_fem_dynamics_object(bodies)
    for attr in ["X", "E", "m", "fext", "lamegU", "wgU", "GNegU", "v"]:
        a, b = np.asarray(getattr(fem, attr), dtype=float), np.asarray(getattr(dump_fem, attr), dtype=float)
        if a.shape != b.shape or not np.allclose(a, b, rtol=1e-5, atol=1e-6 * max(1.0, np.abs(b).max())):
            raise ValueError(f"Rebuilt FEM differs from the Dump in {attr}")


def main():
    parser = argparse.ArgumentParser(description="Convert an editor Dump into a scene file.")
    parser.add_argument("dump", help="Dump file (with a `fem` group)")
    parser.add_argument("scene", help="Output scene file")
    parser.add_argument("--name", default=None, help="Body name prefix (default: from the file name)")
    args = parser.parse_args()
    dump_to_scene(args.dump, args.scene, args.name)
    print(f"Wrote {args.scene}")


if __name__ == "__main__":
    main()
