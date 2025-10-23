# type: ignore
import h5py
import numpy as np
import polyscope as ps
import polyscope.imgui as imgui
import polyscope.implot as implot
import tkinter as tk
from tkinter import filedialog


def _find_frames_group(h5: h5py.File):
    # Prefer 'frames', fallback to 'frame'
    if "frames" in h5:
        return h5["frames"]
    if "frame" in h5:
        return h5["frame"]
    return None


def _sorted_frame_keys(frames_grp: h5py.Group):
    # Expect keys like 'frames/000123' or 'frame/000123' or just '000123'
    keys = list(frames_grp.keys())
    # NOTE: The serializer should already write zero-padded keys, so they are lexicographically sorted
    return keys
    # return sorted(keys, key=lambda k: int(str(k).split("/")[1]))


def _find_integrate_group(frame_grp: h5py.Group):
    # Check all known integrate group names
    names = [
        "pbat.sim.algorithm.vbd.Integrate",
        "pbat.sim.algorithm.vbd.Anderson.Integrate",
        "pbat.sim.algorithm.vbd.Broyden.Integrate",
        "pbat.sim.algorithm.vbd.Chebyshev.Integrate",
    ]
    for name in names:
        if name in frame_grp:
            return name, frame_grp[name]
    return None, None


def _solve_group_name_from_integrate_name(integrate_name: str) -> str:
    # Replace the trailing 'Integrate' with 'Solve'
    return integrate_name.replace("Integrate", "Solve")


def _read_mesh_and_state(integrate_grp: h5py.Group):
    # Fem group and datasets
    fem_path = "pbat.sim.dynamics.FemElastoDynamics"
    mesh_path = f"{fem_path}/pbat.fem.Mesh"
    if fem_path not in integrate_grp:
        return None, None, None
    fem_grp = integrate_grp[fem_path]
    # Read mesh connectivity and positions
    X = None
    E = None
    if mesh_path in integrate_grp:
        mesh_grp = integrate_grp[mesh_path]
    else:
        # Fallback: nested under fem_grp
        mesh_grp = fem_grp.get("pbat.fem.Mesh", None)
    if mesh_grp is not None:
        if "X" in mesh_grp and "E" in mesh_grp:
            X = np.array(mesh_grp["X"])  # shape (3, n)
            E = np.array(mesh_grp["E"])  # shape (4, m)
    # Deformed positions x at this frame
    x = np.array(fem_grp["x"]) if "x" in fem_grp else None
    return X, E, x


def _read_solver_metrics(integrate_grp: h5py.Group, integrate_name: str):
    solve_name = _solve_group_name_from_integrate_name(integrate_name)
    if solve_name not in integrate_grp:
        return [], []
    solve_grp = integrate_grp[solve_name]
    # Iteration groups are zero-padded strings; gather and sort
    it_keys = sorted(list(solve_grp.keys()))
    f_vals = []
    gnorm_vals = []
    for k in it_keys:
        g = solve_grp[k]
        f = g.attrs.get("f", None)
        gnorm = g.attrs.get("gnorm", None)
        if f is not None:
            f_vals.append(f)
        if gnorm is not None:
            gnorm_vals.append(gnorm)
    return f_vals, gnorm_vals


def main():
    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("VBD Simulation Browser")
    ps.init()

    state = {
        "file": None,  # h5py.File or None
        "path": "",  # path to HDF5 file
        "frames_grp": None,  # h5py.Group or None
        "frame_keys": [],  # sorted list of frame keys
        "frame_idx": 0,  # index into frame_keys
        "solver_name": "",  # name of integrate group used
        "vm": None,  # polyscope VolumeMesh handle
        "last_mesh_key": None,  # to detect topology changes
        "f_vals": [],
        "gnorm_vals": [],
    }

    def _load_file():
        # Open file dialog
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Open HDF5 simulation file",
            defaultextension=".h5",
            filetypes=[("HDF5 files", "*.h5;*.hdf5"), ("All files", "*.*")],
        )
        try:
            if file_path:
                if state["file"] is not None:
                    try:
                        state["file"].close()
                    except Exception:
                        pass
                h5 = h5py.File(file_path, "r")
                frames_grp = _find_frames_group(h5)
                frame_keys = (
                    _sorted_frame_keys(frames_grp) if frames_grp is not None else []
                )
                state["file"] = h5
                state["path"] = file_path
                state["frames_grp"] = frames_grp
                state["frame_keys"] = frame_keys
                state["frame_idx"] = 0
                state["solver_name"] = ""
                state["f_vals"] = []
                state["gnorm_vals"] = []
                state["last_mesh_key"] = None
        finally:
            root.destroy()

    def _update_frame():
        # Load selected frame, update mesh and metrics
        h5 = state["file"]
        frames_grp = state["frames_grp"]
        if h5 is None or frames_grp is None or len(state["frame_keys"]) == 0:
            return
        k = state["frame_keys"][state["frame_idx"]]
        if k not in frames_grp:
            return
        frame_grp = frames_grp[k]
        integrate_name, integrate_grp = _find_integrate_group(frame_grp)
        state["solver_name"] = integrate_name or ""
        if integrate_grp is None:
            return
        # Read mesh and state
        X, E, x = _read_mesh_and_state(integrate_grp)
        if (X is None) or (E is None):
            return
        # Prepare positions (prefer x if present)
        V = (x if x is not None else X).T  # to shape (n,3)
        C = E.T  # to shape (m,4)
        # Detect topology changes using a simple key
        mesh_key = (V.shape[0], C.shape[0])
        if state["vm"] is None or state["last_mesh_key"] != mesh_key:
            # Re-register volume mesh
            try:
                state["vm"] = ps.register_volume_mesh("Mesh", V, C)
            except Exception as e:
                print(f"Polyscope register failed: {e}")
                state["vm"] = None
            state["last_mesh_key"] = mesh_key
        else:
            try:
                state["vm"].update_vertex_positions(V)
            except Exception:
                # Fallback to re-registering
                try:
                    state["vm"] = ps.register_volume_mesh("Mesh", V, C)
                except Exception:
                    state["vm"] = None
        # Read solver metrics
        f_vals, gnorm_vals = _read_solver_metrics(integrate_grp, integrate_name)
        state["f_vals"] = f_vals
        state["gnorm_vals"] = gnorm_vals

    def callback():
        imgui.Text("VBD Simulation Browser")
        if imgui.TreeNode("I/O"):
            if imgui.Button("Load HDF5", [imgui.GetWindowWidth() / 2.1, 0]):
                _load_file()
                _update_frame()
            if state["path"]:
                imgui.SameLine()
                imgui.Text(state["path"])
            imgui.TreePop()

        # Frame info and selection
        n_frames = len(state["frame_keys"])
        imgui.Text(f"Frames: {n_frames}")
        if n_frames > 0:
            changed, idx = imgui.SliderInt("Frame", state["frame_idx"], 0, n_frames - 1)
            if changed:
                state["frame_idx"] = idx
                _update_frame()
            # Solver name
            if state["solver_name"]:
                imgui.Text(f'Solver: {state["solver_name"]}')

        # Plots
        if len(state["f_vals"]) > 0:
            if implot.BeginPlot("Objective f per-iteration"):
                implot.SetupAxes("Iteration", "f", implot.ImPlotAxisFlags_None, implot.ImPlotAxisFlags_AutoFit)
                implot.PlotLine(
                    "f",
                    np.arange(len(state["f_vals"])),
                    np.array(state["f_vals"]),
                )
                implot.EndPlot()
        if len(state["gnorm_vals"]) > 0:
            if implot.BeginPlot("Gradient norm per-iteration"):
                implot.SetupAxes("Iteration", "||g||", implot.ImPlotAxisFlags_None, implot.ImPlotAxisFlags_AutoFit)
                implot.PlotLine(
                    "||g||",
                    np.arange(len(state["gnorm_vals"])),
                    np.array(state["gnorm_vals"]),
                )
                implot.EndPlot()

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
