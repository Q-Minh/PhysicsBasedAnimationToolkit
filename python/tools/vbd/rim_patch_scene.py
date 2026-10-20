# type: ignore
"""
Write a scene of a plate (in the xy-plane) held by fixed patches on its rim, e.g. the benchmark trampolines.

The plate's mesh is the first body of a source scene. The output has a homogeneous material (or one Young's
modulus per region, with --regions-from and --region-youngs), gravity along z, and one Fixed transform, active
forever, on the rim nodes (nodes of sideways boundary faces) within --half-angle degrees of --patches evenly
spaced angles around the plate's center, starting at angle 0.

Usage (from the repo root):
    python -m python.tools.vbd.rim_patch_scene source.h5 scene.h5 --youngs 1e6 [--patches 4] [--half-angle 4]         [--regions-from regions.h5 --region-youngs 2=1e8]

The benchmark trampolines (data/scenarios), from the paper's circle trampoline mesh
(path/to/accelerated-vbd-extras/final_scenes/trampolines; the circle files share one mesh and one region layout:
region 1 is the outer ring, r > 1, and region 2 the center, r < 1):
    S=path/to/accelerated-vbd-extras/final_scenes/trampolines/trampoline_circle_stiff-soft.h5
    for Y in 1e5 1e6 1e7 1e8; do
        python -m python.tools.vbd.rim_patch_scene $S data/scenarios/trampoline_circle_Y$Y.h5 --youngs $Y
    done
    # Heterogeneous: stiff 1e8 center, soft 1e6 outer ring
    python -m python.tools.vbd.rim_patch_scene $S data/scenarios/trampoline_circle_het.h5 --youngs 1e6         --regions-from $S --region-youngs 2=1e8
"""

import argparse
import h5py as h5
import numpy as np
from pbatoolkit import pbat
from .ui.tetrahedral_elastodynamics_body import TetrahedralElastodynamicsBody
from .ui.utils import transform_library as tl

BODY = "fem_tet_elastic_bodies/0/tools.vbd.ui.TetrahedralElastodynamicsBody"


def rim_nodes(V, T):
    """Nodes of boundary faces facing sideways (the plate lies in the xy-plane)."""
    _, F = pbat.geometry.simplex_mesh_boundary(T.T, n=V.shape[0])
    F = F.T
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    side = np.abs(n[:, 2]) < 0.5 * np.linalg.norm(n, axis=1)
    return np.unique(F[side])


def patches(V, rim, n_patches, half_angle):
    """Rim nodes within half_angle (degrees) of n_patches evenly spaced angles, starting at 0, per patch."""
    c = V[:, :2].mean(axis=0)
    a = np.degrees(np.arctan2(V[rim, 1] - c[1], V[rim, 0] - c[0]))
    out = []
    for k in range(n_patches):
        d = np.abs((a - k * 360.0 / n_patches + 180.0) % 360.0 - 180.0)
        out.append(rim[d <= half_angle])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument(
        "--youngs", type=float, required=True, help="Young's modulus of the elements not set by --region-youngs"
    )
    parser.add_argument(
        "--regions-from", help="Scene with the same mesh (first body) whose per-element region ids to copy"
    )
    parser.add_argument(
        "--region-youngs", nargs="+", default=[], metavar="R=Y", help="Young's modulus Y of region id R"
    )
    parser.add_argument("--poisson", type=float, default=0.45)
    parser.add_argument("--density", type=float, default=1000.0)
    parser.add_argument("--gravity", type=float, default=-9.81)
    parser.add_argument("--patches", type=int, default=4)
    parser.add_argument("--half-angle", type=float, default=4.0, help="Patch half-width in degrees")
    args = parser.parse_args()
    with h5.File(args.source, "r") as f:
        name = str(f[BODY].attrs["name"])
        V, T = f[BODY]["V"][()].astype(float), f[BODY]["T"][()].astype(np.int64)
    n_elements = T.shape[0]
    R, Ye = None, np.full(n_elements, args.youngs)
    if args.regions_from is not None:
        with h5.File(args.regions_from, "r") as f:
            same = f[BODY]["V"].shape == V.shape and f[BODY]["T"].shape == T.shape
            if not (same and np.array_equal(f[BODY]["V"][()], V) and np.array_equal(f[BODY]["T"][()], T)):
                raise ValueError(f"{args.regions_from}: its first body's mesh differs from {args.source}'s")
            R = f[BODY]["R"][()].astype(np.int64)
    elif args.region_youngs:
        raise ValueError("--region-youngs needs --regions-from")
    for item in args.region_youngs:
        r, y = item.split("=")
        Ye[R == int(r)] = float(y)
    rim = rim_nodes(V, T)
    groups = patches(V, rim, args.patches, args.half_angle)
    fixed = np.sort(np.concatenate(groups))
    body = TetrahedralElastodynamicsBody()
    body.on_mesh_loaded(
        name, V, T, R, Ye, np.full(n_elements, args.poisson),
        np.full(n_elements, args.density), np.zeros((n_elements, 3)), np.array([0.0, 0.0, args.gravity]),
        np.zeros_like(V), headless=True,
    )
    library = tl.TransformLibrary()
    library.on_mesh_added(name)
    fix = tl.FixedTransform("Rim patches", 0.0, np.inf)
    fix.adjust()
    fix._mesh_dirichlet_nodes[name] = fixed
    fix.id = 1
    library.transforms.append(fix)
    with h5.File(args.output, "w") as f:
        grp = f.create_group("fem_tet_elastic_bodies")
        grp.attrs["num_tet_elastic_bodies"] = 1
        body.serialize(grp.create_group("0"))
        f.create_group("static_mesh_colliders").attrs["num_static_mesh_colliders"] = 0
        library.serialize(f.create_group("transform_library"))
        f["recycled_tet_elastic_body_indices"] = np.zeros(0, dtype=int)
    print(
        f"Wrote {args.output}: {V.shape[0]} nodes, {n_elements} elements, {rim.size} rim nodes, "
        f"{fixed.size} fixed ({fixed.size / rim.size:.1%} of the rim) in patches of {[g.size for g in groups]}"
    )
    if R is not None:
        for r in np.unique(R):
            print(f"  region {r}: {(R == r).sum()} elements, Young's modulus {np.unique(Ye[R == r])}")


if __name__ == "__main__":
    main()
