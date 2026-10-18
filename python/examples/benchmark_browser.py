"""Browses a benchmark dataset written by make_benchmark.py in Polyscope.

Pick a problem set (static, or dynamic with time step h), a total energy E and a problem, and see:

    start state x       the problem's initial shape, colored by displacement |x - X| or by det(F) (red/blue
                        diverging around 1; values <= 0 are inverted elements),
    rest shape X        a transparent ghost: the static problems' solution,
    velocity v          the start velocity J(q) qdot as vectors (dynamic problems' samples/v; for static problems
                        it's reconstructed from the shared microcanonical sample),
    inertial target x~  dynamic problems only: x + h v + h^2 aext,
    reference x*        dynamic problems only: the reference solution of the BDF1 step,

with the problem's energies (E, U, K = E - U), inversion diagnostics and reference-solve statistics, and the pool's
quality (split R-hat, ESS) and distribution of U / E, on which the selected problem is marked.

The mesh connectivity isn't stored in the problem files, so it's loaded from the scene recorded in their `input`
attribute (filepath:h5path of a stored FemElastoDynamics), or from --input.

Usage:
    python benchmark_browser.py -d C:\\git\\PhysicsBasedAnimationToolkit\\data\\benchmarks\\trampoline-sample
"""

import argparse
import glob
import os
import re
import typing

import h5py
import numpy as np
import polyscope as ps
import polyscope.imgui as imgui

import make_benchmark
import modalrom


def scan_dataset(root: str) -> typing.Dict[str, typing.List[typing.Tuple[float, str]]]:
    """Finds the problem files of a dataset.

    Args:
        root (str): Dataset directory, as passed to make_benchmark.py's --output.

    Returns:
        typing.Dict[str, typing.List[typing.Tuple[float, str]]]: Problem set name ("static", "dynamic_h{h}") to its
        (total energy, file path) pairs sorted by energy. Static comes first, then dynamic sets by increasing h.
    """
    sets = {}
    for directory in sorted(glob.glob(os.path.join(root, "*"))):
        name = os.path.basename(directory)
        if name != "static" and not name.startswith("dynamic_h"):
            continue
        files = []
        for path in glob.glob(os.path.join(directory, "E*.h5")):
            match = re.fullmatch(r"E(.+)\.h5", os.path.basename(path))
            if match:
                files.append((float(match.group(1)), path))
        if files:
            sets[name] = sorted(files)
    order = lambda name: (name != "static", float(name[len("dynamic_h") :]) if name != "static" else 0.0)
    return {name: sets[name] for name in sorted(sets, key=order)}


def load_problems(path: str) -> dict:
    """Loads the arrays and attributes of a problem file, adding full-space start velocities for static files.

    Args:
        path (str): Problem file (static/E*.h5 or dynamic_h*/E*.h5).

    Returns:
        dict: attrs (root attributes), X (N, 3), x (S, N, 3), energy (S,), min_J (S,) or None, and v (S, N, 3) or None;
        for dynamic files also xtilde, x_ref (S, N, 3), ref_newton_iterations, ref_newton_correction, ref_min_J (S,).
    """
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        g = f["samples"]
        data = dict(
            attrs=attrs,
            X=f["X"][:].astype(np.float64),
            x=g["x"][:].astype(np.float64),
            energy=g["energy"][:],
            min_J=g["min_J"][:] if "min_J" in g else None,
            v=None,
        )
        if "v" in g:
            data["v"] = g["v"][:].astype(np.float64)
        elif "qdot" in g:
            # J(q) qdot with J = U + Q:q, from the stored basis
            U, Q = f["basis/U"][:], f["basis/Q"][:]
            n, m = U.shape
            q, qdot = g["q"][:], g["qdot"][:]
            v = []
            for qs, qds in zip(q, qdot):
                J = U if Q.size == 0 else U + (Q.reshape(n * m, m) @ qs).reshape(n, m)
                v.append((J @ qds).reshape(-1, 3))
            data["v"] = np.array(v)
        for name in ["xtilde", "x_ref"]:
            if name in g:
                data[name] = g[name][:].astype(np.float64)
        for name in ["ref_newton_iterations", "ref_newton_correction", "ref_min_J"]:
            if name in g:
                data[name] = g[name][:]
    return data


def pool_energy_shares(root: str, pool_file: str, energy: float) -> np.ndarray | None:
    """U / E of every pooled sample, from the pool file a problem file refers to, or None if it's missing."""
    path = os.path.join(root, pool_file)
    if not os.path.exists(path):
        return None
    with h5py.File(path, "r") as f:
        return f["samples/energy"][:] / energy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Benchmark dataset browser",
        description="Explore the problems of a dataset written by make_benchmark.py. See the module docstring.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help="Dataset directory (make_benchmark.py's --output), containing static/ and dynamic_h*/",
        type=str,
        dest="dataset",
        required=True,
    )
    parser.add_argument(
        "--input",
        help="filepath:h5path of the stored FemElastoDynamics to take the mesh from. Defaults to the problem files' "
        "input attribute",
        type=str,
        dest="input",
        default=None,
    )
    args = parser.parse_args()

    sets = scan_dataset(args.dataset)
    if not sets:
        parser.error(f"No static/ or dynamic_h*/ problem files in {args.dataset}")
    set_names = list(sets)
    first = load_problems(sets[set_names[0]][0][1])
    scene_input = args.input or str(first["attrs"]["input"])
    try:
        fem = make_benchmark.load_scene(scene_input)
    except (ValueError, OSError) as error:
        parser.error(f"Can't load the mesh from {scene_input} ({error}); pass --input")
    E, X = np.asarray(fem.E), np.asarray(fem.X)
    if np.abs(X.T - first["X"]).max() > 1e-5:
        parser.error(f"The rest positions of {scene_input} don't match the dataset's X")
    egU, GNegU = fem.egU, fem.GNegU
    dirichlet = np.asarray(fem.dirichlet_nodes)

    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.init()
    start = ps.register_volume_mesh("start state x", X.T, E.T)
    rest = ps.register_volume_mesh("rest shape X", X.T, E.T, transparency=0.2, color=(0.6, 0.6, 0.6))
    target = ps.register_volume_mesh(
        "inertial target x~", X.T, E.T, transparency=0.35, color=(0.95, 0.6, 0.2), enabled=False
    )
    reference = ps.register_volume_mesh(
        "reference x*", X.T, E.T, transparency=0.5, color=(0.2, 0.75, 0.4), enabled=False
    )
    if dirichlet.size:
        ps.register_point_cloud("Dirichlet nodes", X.T[dirichlet], radius=0.003, color=(0.1, 0.1, 0.1))

    colorings = ["Displacement |x - X| (m)", "det F"]
    state = dict(set_idx=0, energy_idx=0, problem=0, coloring=0, data=None, key=None, shares=None)

    def select():
        """Loads the selected file if it changed and updates the meshes for the selected problem."""
        name = set_names[state["set_idx"]]
        state["energy_idx"] = min(state["energy_idx"], len(sets[name]) - 1)
        energy, path = sets[name][state["energy_idx"]]
        if state["key"] != path:
            state["data"] = load_problems(path)
            state["key"] = path
            state["shares"] = pool_energy_shares(
                args.dataset, str(state["data"]["attrs"].get("pool_file", "")), energy
            )
        data = state["data"]
        state["problem"] = min(state["problem"], data["x"].shape[0] - 1)
        s = state["problem"]
        x = data["x"][s]
        start.update_vertex_positions(x)
        if state["coloring"] == 0:
            start.add_scalar_quantity(
                "color", np.linalg.norm(x - data["X"], axis=1), defined_on="vertices", enabled=True, cmap="viridis"
            )
        else:
            detF = modalrom.deformation_gradient_determinants(E, egU, GNegU, x.ravel())
            start.add_scalar_quantity(
                "color", detF, defined_on="cells", enabled=True, cmap="coolwarm", vminmax=(0.0, 2.0)
            )
        if data["v"] is not None:
            start.add_vector_quantity("velocity v", data["v"][s], defined_on="vertices", enabled=True)
        dynamic = "xtilde" in data
        if dynamic:
            target.update_vertex_positions(data["xtilde"][s])
            reference.update_vertex_positions(data["x_ref"][s])
        else:
            target.set_enabled(False)
            reference.set_enabled(False)

    def callback():
        changed_any = False
        changed, state["set_idx"] = imgui.Combo("Problem set", state["set_idx"], set_names)
        changed_any |= changed
        name = set_names[state["set_idx"]]
        labels = [f"E = {energy:g} J" for energy, _ in sets[name]]
        state["energy_idx"] = min(state["energy_idx"], len(labels) - 1)
        changed, state["energy_idx"] = imgui.Combo("Total energy", state["energy_idx"], labels)
        changed_any |= changed
        if state["data"] is None or changed_any:
            select()
        data = state["data"]
        S = data["x"].shape[0]
        changed, state["problem"] = imgui.SliderInt("Problem", state["problem"], 0, S - 1)
        changed_c, state["coloring"] = imgui.Combo("Color start state by", state["coloring"], colorings)
        if changed or changed_c:
            select()

        attrs, s = data["attrs"], state["problem"]
        energy = float(attrs["target_energy"])
        U = float(data["energy"][s])
        imgui.Separator()
        imgui.TextWrapped(str(attrs.get("problem_definition", "")))
        imgui.Text(f"E = {energy:g} J: U = {U:.4g} J, K = E - U = {energy - U:.4g} J (U/E = {U / energy:.3f})")
        max_disp = np.linalg.norm(data["x"][s] - data["X"], axis=1).max()
        imgui.Text(f"Peak displacement |x - X| = {max_disp:.4g} m")
        if data["min_J"] is not None:
            min_J = float(data["min_J"][s])
            text = f"Min det F = {min_J:.3f}"
            if min_J <= 0:
                imgui.TextColored((1.0, 0.4, 0.3, 1.0), text + " (inverted elements)")
            else:
                imgui.Text(text)
        if "xtilde" in data:
            h = float(attrs["h"])
            v = data["v"][s]
            step = np.linalg.norm(data["x_ref"][s] - data["x"][s], axis=1).max()
            imgui.Text(f"h = {h:g} s: |h v| max {h * np.linalg.norm(v, axis=1).max():.3g} m, |x* - x| max {step:.3g} m")
            imgui.Text(
                f"Reference: {int(data['ref_newton_iterations'][s])} Newton iterations, last correction "
                f"{float(data['ref_newton_correction'][s]):.2g} m, min det F {float(data['ref_min_J'][s]):.3f}"
            )
            target.set_enabled(imgui.Checkbox("Show inertial target x~", target.is_enabled())[1])
            reference.set_enabled(imgui.Checkbox("Show reference x*", reference.is_enabled())[1])
        rest.set_enabled(imgui.Checkbox("Show rest shape X", rest.is_enabled())[1])

        imgui.Separator()
        rhat = attrs.get("rhat")
        rhat_text = "n/a" if rhat is None else f"{float(rhat):.3f}" + ("" if rhat < 1.05 else " (>= 1.05)")
        imgui.Text(
            f"Pool: split R-hat {rhat_text}, ESS {float(attrs.get('ess', 0)):.0f}, "
            f"problem {s} is pool sample {int(attrs['pool_indices'][s])}"
        )
        if state["shares"] is not None:
            counts, edges = np.histogram(state["shares"], bins=24, range=(0.0, 1.0))
            imgui.PlotHistogram(
                "U / E over the pool",
                counts.astype(np.float32).tolist(),
                overlay_text=f"this problem: U/E = {U / energy:.3f} (bin {min(int(U / energy * 24), 23) + 1}/24)",
                scale_min=0.0,
                graph_size=(0.0, 80.0),
            )
        problems = data["energy"] / energy
        imgui.Text("U/E of this file's problems: " + ", ".join(f"{share:.2f}" for share in problems))

    ps.set_user_callback(callback)
    ps.show()
