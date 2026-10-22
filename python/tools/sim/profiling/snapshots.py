"""
Snapshots of a ``simulate.py`` trajectory for paper figures (e.g. the profiling timeline's snapshot row):
Polyscope renders of the saved frames nearest to given simulated times, from one fixed camera, on a white
background without a ground plane, each body type in its own color, all cropped to the same size.

Bodies are the FEM's connected components; bodies with the same number of nodes count as one type (e.g. the
capsules, the tubes and the bowl of the capsule bowl) and share a color, assigned in node order from
``--colors``. Only the boundary surface of the tetrahedra is drawn.

The camera is a Polyscope view JSON file (``--view``). ``--pick`` opens the viewer instead, with a frame slider
and a button that saves the current view to ``--view``, so the camera can be chosen interactively and
reused. Without a view file, the camera is ``--look-at`` (camera position and target), else Polyscope's home
view of the first frame, and is saved to ``--view`` if given.

Polyscope has no headless backend on Windows: rendering opens a window briefly.

Usage (from the repo root):

    python -m python.tools.sim.profiling.snapshots out.h5 --times 0.9 1.4 2.9 --view view.json -o out/snap \
        [--pick] [--size 1200 1200] [--colors "#e69f00" "#9db4c0" "#56b4e9"]

Writes ``<prefix>_<i>.png`` (i = 1, 2, ... in time order) and prints each image's frame time.
"""

import argparse
import os

import h5py
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

# Colors per body type, in node order: muted Okabe-Ito hues, distinct from the profiling phase colors
DEFAULT_COLORS = ("#e69f00", "#9db4c0", "#56b4e9", "#cc79a7", "#009e73", "#f0e442")
CROP_PAD = 0.02  # fraction of the crop's size added around the union of the images' content


def load_frames(h5f: h5py.File) -> tuple[list[str], np.ndarray]:
    """The trajectory's frame groups in order and their simulated times (s), from their step counts."""
    frames = sorted(k for k in h5f["sim"].keys() if k.isdigit())
    dt = float(h5f["params/Integration"].attrs["dt"])
    times = np.array([int(h5f["sim"][k].attrs.get("t", 0)) * dt for k in frames])
    return frames, times


def body_types(n_nodes: int, E: np.ndarray) -> np.ndarray:
    """Per node, its body type: connected components with the same node count share a type, numbered in
    order of their first node."""
    rows = np.repeat(E, 4, axis=1).ravel()
    cols = np.tile(E, (1, 4)).ravel()
    graph = sp.coo_matrix((np.ones_like(rows, dtype=np.int8), (rows, cols)), shape=(n_nodes, n_nodes))
    _, component = connected_components(graph, directed=False)
    sizes = np.bincount(component)
    node_size = sizes[component]
    _, first = np.unique(node_size, return_index=True)
    order = np.argsort(first)  # sizes ordered by their first node
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return rank[np.searchsorted(np.unique(node_size), node_size)]


def boundary_faces(E: np.ndarray) -> np.ndarray:
    """The triangles that belong to exactly one tetrahedron."""
    faces = np.concatenate([E[:, [1, 2, 3]], E[:, [0, 3, 2]], E[:, [0, 1, 3]], E[:, [0, 2, 1]]])
    keys = np.sort(faces, axis=1)
    _, index, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    return faces[index[counts == 1]]


def crop_box(images: list[np.ndarray], background: float = 1.0) -> tuple[int, int, int, int]:
    """The union over ``images`` of the pixels that aren't background, padded by :data:`CROP_PAD`:
    (row0, row1, col0, col1)."""
    mask = np.zeros(images[0].shape[:2], dtype=bool)
    for image in images:
        mask |= np.any(np.abs(image[..., :3] - background) > 1e-3, axis=2)
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    pad = int(CROP_PAD * max(np.ptp(rows), np.ptp(cols)))
    return (max(rows.min() - pad, 0), min(rows.max() + pad + 1, mask.shape[0]),
            max(cols.min() - pad, 0), min(cols.max() + pad + 1, mask.shape[1]))


def _rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) / 255 for i in (0, 2, 4))


def main():
    parser = argparse.ArgumentParser(description="Polyscope snapshots of a simulate.py trajectory.")
    parser.add_argument("trajectory", help="simulate.py output (.h5).")
    parser.add_argument("--times", type=float, nargs="*", default=[], help="Simulated times (s) to render.")
    parser.add_argument("-o", "--output", default=None, help="Output prefix: writes <prefix>_<i>.png.")
    parser.add_argument("--view", default=None, help="Polyscope view JSON file (read, or written by --pick).")
    parser.add_argument("--pick", action="store_true", help="Open the viewer to choose and save the camera.")
    parser.add_argument("--look-at", type=float, nargs=6, default=None, metavar=("X", "Y", "Z", "TX", "TY", "TZ"),
                        dest="look_at", help="Camera position and target, without a view file.")
    parser.add_argument("--size", type=int, nargs=2, default=(1200, 1200), metavar=("W", "H"),
                        help="Window size in pixels before cropping.")
    parser.add_argument("--colors", nargs="*", default=list(DEFAULT_COLORS),
                        help="Colors per body type, in node order.")
    parser.add_argument("--translucent", type=int, nargs="*", default=[],
                        help="Body types (indices in node order) drawn translucent, e.g. containers that hide "
                        "what is inside them.")
    parser.add_argument("--opacity", type=float, default=0.35, help="Opacity of the --translucent types.")
    args = parser.parse_args()

    import polyscope as ps
    import polyscope.imgui as psim

    with h5py.File(args.trajectory, "r") as f:
        E = np.asarray(f["FemElastoDynamics/E"], dtype=np.int64)
        X = np.asarray(f["FemElastoDynamics/X"], dtype=np.float64)
        frames, times = load_frames(f)
        positions = lambda i: np.asarray(f["sim"][frames[i]]["x"], dtype=np.float64)  # noqa: E731
        types = body_types(X.shape[0], E)
        faces = boundary_faces(E)
        face_type = types[faces[:, 0]]

        ps.set_program_name("Snapshots")
        ps.set_up_dir("z_up")
        ps.set_front_dir("neg_y_front")
        ps.set_ground_plane_mode("none")
        ps.set_SSAA_factor(4)
        ps.set_background_color((1.0, 1.0, 1.0))
        if args.translucent:
            ps.set_transparency_mode("pretty")
        ps.set_window_size(*args.size)
        ps.init()
        ps.set_window_size(*args.size)
        meshes = []
        x0 = positions(0)
        for k in range(int(types.max()) + 1):
            mesh = ps.register_surface_mesh(f"type {k}", x0, faces[face_type == k],
                                            color=_rgb(args.colors[k % len(args.colors)]), edge_width=0.0)
            mesh.set_back_face_policy("identical")
            if k in args.translucent:
                mesh.set_transparency(args.opacity)
            meshes.append(mesh)
        if args.view and os.path.isfile(args.view):
            with open(args.view, encoding="utf-8") as v:
                ps.set_view_from_json(v.read())
        else:
            if args.look_at:
                ps.look_at(args.look_at[:3], args.look_at[3:])
            if args.view:
                with open(args.view, "w", encoding="utf-8") as v:
                    v.write(ps.get_view_as_json())
                print(f"Wrote {args.view}")

        def show(i: int):
            x = positions(i)
            for mesh in meshes:
                mesh.update_vertex_positions(x)

        if args.pick:
            state = {"frame": 0}

            def callback():
                changed, state["frame"] = psim.SliderInt("Frame", state["frame"], 0, len(frames) - 1)
                psim.Text(f"t = {times[state['frame']]:.3f} s")
                if changed:
                    show(state["frame"])
                if args.view and psim.Button("Save view"):
                    with open(args.view, "w", encoding="utf-8") as v:
                        v.write(ps.get_view_as_json())
                    print(f"Wrote {args.view}")

            ps.set_user_callback(callback)
            ps.show()
            return

        if not args.output or not args.times:
            raise SystemExit("--output and --times are needed to render (or --pick)")
        ps.set_build_default_gui_panels(False)
        images, picked = [], []
        for t in sorted(args.times):
            i = int(np.argmin(np.abs(times - t)))
            show(i)
            ps.frame_tick()
            images.append(ps.screenshot_to_buffer(transparent_bg=False, include_UI=False).astype(np.float64) / 255)
            picked.append(times[i])
    r0, r1, c0, c1 = crop_box(images)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    import matplotlib.pyplot as plt

    for n, (image, t) in enumerate(zip(images, picked), start=1):
        path = f"{args.output}_{n}.png"
        plt.imsave(path, np.clip(image[r0:r1, c0:c1, :3], 0, 1))
        print(f"Wrote {path} (t = {t:.3f} s)")


import unittest


class TestSnapshots(unittest.TestCase):

    def test_types_faces_crop(self):
        # Two single tets (4 nodes each) and one two-tet body (5 nodes)
        E = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [9, 10, 11, 12]])
        types = body_types(13, E)
        self.assertEqual(types.tolist(), [0] * 8 + [1] * 5)
        faces = boundary_faces(E)
        self.assertEqual(len(faces), 4 + 4 + 6)  # the shared face is interior
        a = np.ones((10, 10, 3))
        b = np.ones((10, 10, 3))
        a[2:4, 3:5] = 0.0
        b[6:8, 1:2] = 0.0
        self.assertEqual(crop_box([a, b]), (2, 8, 1, 5))


if __name__ == "__main__":
    main()
