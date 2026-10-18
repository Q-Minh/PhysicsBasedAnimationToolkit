# type: ignore
"""
One-off migration of editor scenes authored before L_TRANSLATE became a per-node radial translation.

The old LocalTranslateTransform divided the radial vectors of its nodes by the Frobenius norm of the
whole group, rho0 = ||[r_1 ... r_n]||_F with r_j the xy-component of x_j - origin. The motion was a
uniform xy-scaling about origin, and node j moved radially at speed * |r_j| / rho0. Now every node moves
radially at `speed`. This script replaces `speed` by the mean of the old per-node speeds
(speed * mean_j |r_j| / rho0, with rho0 per mesh as the editor applied it mesh by mesh). That reproduces
the old motion exactly when all nodes share one radius, as on the rims of the trampoline scenes. The
spread of the old per-node speeds is printed so that inexact migrations are visible.

Usage (from the repo root):
    python -m python.tools.vbd.migrate_l_translate scene.h5 [scene2.h5 ...] [--dry-run]

Each modified file is first copied to `<scene>.pre-l-translate-migration.h5`. A file that already has
the `l_translate_migrated` attribute on its transform library is skipped.
"""

import argparse
import shutil
import h5py
import numpy as np

_TLIB = "transform_library"
_BODIES = "fem_tet_elastic_bodies"
_BODY_TYPE = "tools.vbd.ui.TetrahedralElastodynamicsBody"
_L_TRANSLATE = 4
_MIGRATED_ATTR = "l_translate_migrated"


def _mesh_positions(f: h5py.File) -> dict[str, np.ndarray]:
    """Mesh name -> (n, 3) world positions of the scene's elastic bodies."""
    positions = {}
    for b in range(int(f[_BODIES].attrs["num_tet_elastic_bodies"])):
        body = f[f"{_BODIES}/{b}/{_BODY_TYPE}"]
        positions[str(body.attrs["name"])] = body["V"][()]
    return positions


def migrated_speed(
    speed: float, origin: np.ndarray, mesh_nodes: dict[str, np.ndarray], positions
):
    """Return (new speed, old per-node speeds) for one old-style L_TRANSLATE transform."""
    per_node = []
    for mesh, nodes in mesh_nodes.items():
        if nodes.size == 0:
            continue
        r = positions[mesh][nodes] - origin[np.newaxis, :]
        r[:, 2] = 0
        radii = np.linalg.norm(r, axis=1)
        rho0 = np.linalg.norm(radii)
        if rho0 > 0:
            per_node.append(speed * radii / rho0)
    if len(per_node) == 0:
        return speed, np.array([])
    per_node = np.concatenate(per_node)
    return float(per_node.mean()), per_node


def migrate(path: str, dry_run: bool = False) -> bool:
    with h5py.File(path, "r") as f:
        if _TLIB not in f or "transform" not in f[_TLIB]:
            print(f"{path}: no transforms, skipped")
            return False
        if f[_TLIB].attrs.get(_MIGRATED_ATTR, False):
            print(f"{path}: already migrated, skipped")
            return False
        positions = _mesh_positions(f)
        updates = {}
        for tid, tgrp in f[f"{_TLIB}/transform"].items():
            if int(tgrp.attrs["transform_type"]) != _L_TRANSLATE:
                continue
            mesh_nodes = {
                mesh: mgrp["node_indices"][()]
                for mesh, mgrp in tgrp.get("dirichlet/mesh", {}).items()
            }
            speed = float(tgrp.attrs["speed"])
            new_speed, per_node = migrated_speed(
                speed, np.asarray(tgrp.attrs["origin"], dtype=float), mesh_nodes, positions
            )
            spread = (
                f", old per-node speeds in [{per_node.min():.6g}, {per_node.max():.6g}]"
                if per_node.size > 0
                else ", no nodes"
            )
            print(
                f"{path}: transform {tid} '{tgrp.attrs['name']}': speed {speed:.6g} -> {new_speed:.6g}{spread}"
            )
            updates[tid] = new_speed
    if len(updates) == 0:
        print(f"{path}: no L_TRANSLATE transforms, skipped")
        return False
    if dry_run:
        return False
    backup = path.removesuffix(".h5") + ".pre-l-translate-migration.h5"
    shutil.copy2(path, backup)
    with h5py.File(path, "a") as f:
        for tid, new_speed in updates.items():
            f[f"{_TLIB}/transform/{tid}"].attrs["speed"] = new_speed
        f[_TLIB].attrs[_MIGRATED_ATTR] = True
    print(f"{path}: migrated, backup at {backup}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate L_TRANSLATE speeds of editor scenes to per-node radial translation."
    )
    parser.add_argument("scenes", nargs="+", help="Scene .h5 files to migrate in place.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the new speeds without writing."
    )
    args = parser.parse_args()
    for path in args.scenes:
        migrate(path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
