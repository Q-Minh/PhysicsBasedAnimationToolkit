# type: ignore
import math
import meshio
import numpy as np
import os
import polyscope as ps
import polyscope.imgui as imgui
import tkinter as tk
from tkinter import filedialog

from pbatoolkit import pbat, pypbat


class SceneMesh:
    def __init__(self, name: str, V: np.ndarray, C: np.ndarray):
        self.name = name
        self.V = V  # (n,3)
        self.C = C  # (m,4)
        self.handle = ps.register_volume_mesh(name, V, C)
        # Per-mesh settings (future extension: heterogeneous materials, loads, constraints)
        self.Y = 1e6
        self.nu = 0.45
        self.rho = 1e3
        self.v0 = np.array([0.0, 0.0, 0.0])
        self.b = np.array(
            [0.0, 0.0, 0.0]
        )  # per-mesh body force (e.g. wind, extra load)

    def transformed_vertices(self) -> np.ndarray:
        T = self.handle.get_transform()
        VH = np.vstack([self.V.T, np.ones((1, self.V.shape[0]))])
        VT = (T @ VH).T[:, :3]
        return VT


class SceneState:
    def __init__(self):
        self.meshes: list[SceneMesh] = []
        # Global scene settings (environment/dynamics only)
        self.aext = np.array([0.0, 0.0, -9.81])
        self.dt = 1e-2
        self.s = 1
        # Global Dirichlet convenience (same selection strategy as simple_api)
        self.d_axis = 0
        self.d_percent = 0.01
        self.d_extremity = 0
        # Constructed FEM
        self.fem = None  # pbat.sim.dynamics.FemElastoDynamics

    def add_mesh_from_file(self, file_path: str):
        imesh = meshio.read(file_path)
        if "tetra" in imesh.cells_dict:
            V, C = imesh.points, imesh.cells_dict["tetra"]
            filename = os.path.basename(file_path)
            name = f"{len(self.meshes):<2} - {filename}"
            item = SceneMesh(name, V, C)
            self.meshes.append(item)
        else:
            ps.error("Only tetrahedral meshes are supported in the scene editor.")

    def remove_mesh(self, idx: int):
        if 0 <= idx < len(self.meshes):
            item = self.meshes.pop(idx)
            try:
                ps.remove_volume_mesh(item.name)
            except Exception:
                pass

    def combined_geometry(self):
        if len(self.meshes) == 0:
            return None, None, []
        V_all = []
        C_all = []
        vertex_ranges = []  # list of (start_vertex, end_vertex) for per-mesh indexing
        element_ranges = (
            []
        )  # list of (start_element, end_element) for per-mesh indexing
        v_offset = 0
        e_offset = 0
        for m in self.meshes:
            VT = m.transformed_vertices()
            V_all.append(VT)
            C_all.append(m.C + v_offset)
            n_verts = m.V.shape[0]
            n_elems = m.C.shape[0]
            vertex_ranges.append((v_offset, v_offset + n_verts))
            element_ranges.append((e_offset, e_offset + n_elems))
            v_offset += n_verts
            e_offset += n_elems
        V_all = np.vstack(V_all)
        C_all = np.vstack(C_all)
        return V_all, C_all, vertex_ranges, element_ranges

    def build_fem_elastodynamics(self):
        V_all, C_all, vertex_ranges, element_ranges = self.combined_geometry()
        if V_all is None:
            self.fem = None
            return
        fem = pbat.sim.dynamics.FemElastoDynamics(V_all.T, C_all.T)
        dims = fem.X.shape[0]
        element = pbat.fem.Element.Tetrahedron
        order = 1  # linear shape functions only
        qorder_M = 2 * order
        qorder_U = order
        n_elems = fem.E.shape[1]

        # Mass quadrature (use 2*order)
        wgM = pbat.fem.mesh_quadrature_weights(
            fem.E, fem.X, element, order=order, quadrature_order=qorder_M
        )
        egM = pbat.fem.mesh_quadrature_elements(fem.E, wgM)
        XgM = pbat.fem.mesh_reference_quadrature_points(
            n_elems, element=element, order=order, quadrature_order=qorder_M
        )
        rhog = np.zeros_like(wgM)
        for (start, end), m in zip(element_ranges, self.meshes):
            n_elem_quads = wgM.shape[0]
            rhog[:, start:end] = np.full((n_elem_quads, end - start), m.rho)

        # Elasticity quadrature (use order)
        wgU = pbat.fem.mesh_quadrature_weights(
            fem.E, fem.X, element, order=order, quadrature_order=qorder_U
        )
        egU = pbat.fem.mesh_quadrature_elements(fem.E, wgU)
        XgU = pbat.fem.mesh_reference_quadrature_points(
            n_elems, element=element, order=order, quadrature_order=qorder_U
        )
        mug, lambdag = np.zeros_like(wgU), np.zeros_like(wgU)
        for (start, end), m in zip(element_ranges, self.meshes):
            n_elem_quads = wgU.shape[0]
            mu, llambda = pypbat.fem.lame_coefficients(m.Y, m.nu)
            mug[:, start:end] = np.full((n_elem_quads, end - start), mu)
            lambdag[:, start:end] = np.full((n_elem_quads, end - start), llambda)

        # External load on elasticity quadrature: rho*aext + per-mesh b
        aext = np.asarray(self.aext).reshape(dims, 1)
        fext = (m.rho * aext) + m.b.reshape(dims, 1)  # dims x 1
        bg = np.zeros((dims, math.prod(wgU.shape)))  # dims x # quad.pts.
        for (start, end), m in zip(element_ranges, self.meshes):
            n_elem_quads = wgU.shape[0]
            fext = (m.rho * aext) + m.b.reshape(dims, 1)  # dims x 1
            bg[:, start * n_elem_quads : end * n_elem_quads] = np.repeat(
                fext, n_elem_quads * (end - start), axis=1
            )

        # Apply heterogeneous fields
        fem.set_mass_matrix(
            np.ravel(egM, order="F"),
            np.ravel(wgM, order="F"),
            XgM,
            np.ravel(rhog, order="F"),
        )
        fem.set_elastic_energy(
            np.ravel(egU, order="F"),
            np.ravel(wgU, order="F"),
            XgU,
            np.ravel(mug, order="F"),
            np.ravel(lambdag, order="F"),
        )
        fem.set_external_load(
            np.ravel(egU, order="F"),
            np.ravel(wgU, order="F"),
            XgU,
            bg,
        )
        # Time integration scheme
        fem.set_time_integration_scheme(self.dt, self.s)
        # Set initial conditions
        x0 = V_all.T
        v0 = np.zeros_like(x0)
        for (start, end), m in zip(vertex_ranges, self.meshes):
            v0[:, start:end] = np.repeat(m.v0[:, None], end - start, axis=1)
        fem.set_initial_conditions(x0, v0)
        # Dirichlet constraints
        aabb: pbat.geometry.AxisAlignedBoundingBox3 = pypbat.geometry.aabb(fem.X)
        Xmin, Xmax = aabb.min.copy(), aabb.max.copy()
        extent = Xmax - Xmin
        if self.d_extremity == 0:
            Xmax[self.d_axis] = Xmin[self.d_axis] + self.d_percent * extent[self.d_axis]
            Xmin[self.d_axis] -= self.d_percent * extent[self.d_axis]
        else:
            Xmin[self.d_axis] = Xmax[self.d_axis] - self.d_percent * extent[self.d_axis]
            Xmax[self.d_axis] += self.d_percent * extent[self.d_axis]
        aabb.min, aabb.max = Xmin, Xmax
        d_nodes = aabb.contained(fem.X)
        d_mask = np.zeros(fem.X.shape[1], dtype=bool)
        d_mask[d_nodes] = True
        fem.constrain(d_mask)
        # Set as current
        self.fem = fem

    def serialize_fem(self, path: str):
        if self.fem is None:
            return
        # Use Archive to write scene into HDF5 root
        ac = pbat.io.Archive(path, pbat.io.AccessMode.Overwrite)
        self.fem.serialize(ac)
        ac.flush()


def _load_mesh(state: SceneState):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select tetrahedral mesh file",
        defaultextension=".mesh",
        filetypes=[
            ("Tetrahedral mesh files (ASCII)", "*.mesh"),
            ("Tetrahedral mesh files (binary)", "*.msh"),
            ("All files", "*.*"),
        ],
    )
    try:
        if file_path:
            state.add_mesh_from_file(file_path)
    finally:
        root.destroy()


def _save_fem(state: SceneState):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(
        title="Save FEM scene (HDF5)",
        defaultextension=".h5",
        filetypes=[("HDF5 files", "*.h5;*.hdf5"), ("All files", "*.*")],
    )
    try:
        if file_path:
            state.build_fem_elastodynamics()
            state.serialize_fem(file_path)
    finally:
        root.destroy()


def main():
    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Scene Editor (FEM)")
    ps.init()

    state = SceneState()

    def callback():
        imgui.Text("Scene Editor (FEM)")
        default_button_size = [imgui.GetWindowWidth() / 2.1, 0]

        # Top-level I/O and scene actions
        if imgui.TreeNode("Scene"):
            if imgui.Button("Add Mesh", default_button_size):
                _load_mesh(state)
            if imgui.Button("Save FEM", default_button_size):
                _save_fem(state)
            imgui.Text(f"Meshes: {len(state.meshes)}")
            if state.fem is not None:
                imgui.Text(
                    f"FEM nodes: {state.fem.X.shape[1]}, elements: {state.fem.E.shape[1]}"
                )
            imgui.TreePop()

        # Dynamics (environment-level)
        if imgui.TreeNode("Dynamics"):
            _, state.aext = imgui.InputFloat3("External acceleration", state.aext)
            _, state.dt = imgui.InputFloat("Time step", state.dt)
            _, state.s = imgui.InputInt("BDF step", state.s)
            if imgui.TreeNode("Dirichlet Constraints"):
                _, state.d_axis = imgui.InputInt("Axis (0=x,1=y,2=z)", state.d_axis)
                _, state.d_percent = imgui.InputFloat("Percentage", state.d_percent)
                _, state.d_extremity = imgui.InputInt(
                    "Extremity (0=min,1=max)", state.d_extremity
                )
                imgui.TreePop()
            imgui.TreePop()

        # Per-mesh controls
        for i, m in enumerate(state.meshes):
            if imgui.TreeNode(f"{m.name}"):
                # Material & ICs (not yet heterogeneous in FEM build, but tracked per-mesh)
                if imgui.TreeNode("Material"):
                    _, m.Y = imgui.InputFloat("Young's Modulus", m.Y)
                    _, m.nu = imgui.InputFloat("Poisson's Ratio", m.nu)
                    _, m.rho = imgui.InputFloat("Mass Density", m.rho)
                    imgui.TreePop()
                if imgui.TreeNode("Body Force"):
                    changed, vec = imgui.InputFloat3("b", m.b)
                    if changed:
                        m.b = np.array(vec)
                    imgui.TreePop()
                if imgui.TreeNode("Initial Velocity"):
                    changed, vec = imgui.InputFloat3("v0", m.v0)
                    if changed:
                        m.v0 = np.array(vec)
                    imgui.TreePop()
                if imgui.Button("Delete", default_button_size):
                    state.remove_mesh(i)
                    imgui.TreePop()
                    break  # indices shifted
                imgui.TreePop()

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
