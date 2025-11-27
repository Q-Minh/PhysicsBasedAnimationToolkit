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
from utils.dirichlet_index_library import DirichletIndices, DirichletLibrary
import utils.transform_library as tlib
import python.tools.vbd.utils.box_selection as box_selection
from utils.scene_mesh import SceneMesh

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

        # Library of fixed transforms for Dirichlet groups
        self.transform_library = tlib.TransformLibrary()
        self.transform_library.deserialize("primitive_transforms.h5")

        # Extras when creating new transforms
        self.unsaved_transforms = {}
        for ttype in tlib.TransformType:
            self.unsaved_transforms[ttype] = tlib.PrimitiveTransform.make_default(ttype) 
        self.selected_ttype = 0
        self.editing_new_transform = False
        
        # Library of Dirichlet indices per transform
        self.selected_dirichlet_group = 0
        self.dirichlet_library = {}
        
        # Selection boxes
        self.selections = []

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
                        # add to all Dirichlet libraries
            for tname in self.dirichlet_library:
                self.dirichlet_library[tname].add_mesh(item)
            return item
        else:
            ps.error("Only tetrahedral meshes are supported in the scene editor.")
            return None

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
            fext = (m.rho * aext) + m.b.reshape(dims, 1)  # dims x # quad.pts.
            bg[:, start * n_elem_quads : end * n_elem_quads] = fext

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
        # Default axis picking disabled for now, but should come back as UI element!
        # aabb: pbat.geometry.AxisAlignedBoundingBox3 = pypbat.geometry.aabb(fem.X)
        # Xmin, Xmax = aabb.min.copy(), aabb.max.copy()
        # extent = Xmax - Xmin
        # if self.d_extremity == 0:
        #     Xmax[self.d_axis] = Xmin[self.d_axis] + self.d_percent * extent[self.d_axis]
        #     Xmin[self.d_axis] -= self.d_percent * extent[self.d_axis]
        # else:
        #     Xmin[self.d_axis] = Xmax[self.d_axis] - self.d_percent * extent[self.d_axis]
        #     Xmax[self.d_axis] += self.d_percent * extent[self.d_axis]
        # aabb.min, aabb.max = Xmin, Xmax
        # d_nodes = aabb.contained(fem.X)
        # d_mask = np.zeros(fem.X.shape[1], dtype=bool)
        # d_mask[d_nodes] = True
        d_mask = np.zeros(fem.X.shape[1], dtype=int)
        for (start, end), m in zip(element_ranges, self.meshes):
            for t in self.transform_library.transforms:
                dgroup = self.dirichlet_library[t.name].get_mesh_indices(m)
                if dgroup.indices is not None and len(dgroup.indices) > 0:
                    d_mask[dgroup.indices + start] = t.id
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

    def save_transform_library(self):
        self.transform_library.serialize("primitive_transforms.h5")


    def load_transform_file(self):
        self.transform_library.deserialize("primitive_transforms.h5")
        # We will have one point cloud per transform. This point cloud is stored in the DirichletLibrary
        for t in self.transform_library.transforms:
            self.dirichlet_library[t.name] = DirichletLibrary(t.name)
            for m in self.meshes:
                self.dirichlet_library[t.name].add_mesh(m)

    def add_transform(self, transform: tlib.PrimitiveTransform):
        self.transform_library.add_transform(transform)
        self.dirichlet_library[transform.name] = DirichletLibrary(transform.name)
        for m in self.meshes:
            self.dirichlet_library[transform.name].add_mesh(m)
        print(self.transform_library.transforms)
        print(self.dirichlet_library.keys())

    def spawn_box_selection(self):
        name = f"Selection Box {len(self.selections)}"
        box = box_selection.BoxSelection(name)
        self.selections.append(box)
        return box
    
    def get_box_selection(self, idx: int):
        if 0 <= idx < len(self.selections):
            return self.selections[idx]
        return None


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
            state.save_transform_library()
    finally:
        root.destroy()


def transform_editor(transform: tlib.PrimitiveTransform, idx: int):
    imgui.PushID(idx)
    _, transform.name = imgui.InputText("Name", transform.name)
    _, transform.begin = imgui.InputFloat("Begin Time", transform.begin)
    _, transform.duration = imgui.InputFloat("Duration", transform.duration)

    if transform.transform_type == tlib.TransformType.G_ROTATE:
        _, axis = imgui.InputFloat3("Axis", transform.axis)
        transform.axis = np.array(axis)
        if imgui.Button("Normalize"):
            transform.adjust()
        _, transform.degrees_per_second = imgui.InputFloat("Degrees per Second", transform.degrees_per_second)
    
    elif transform.transform_type == tlib.TransformType.L_ROTATE:
        _, axis = imgui.InputFloat3("Axis", transform.axis)
        transform.axis = np.array(axis)
        if imgui.Button("Normalize"):
            transform.adjust()
        _, origin = imgui.InputFloat3("Origin", transform.origin)
        transform.origin = np.array(origin)
        _, transform.degrees_per_second = imgui.InputFloat("Degrees per Second", transform.degrees_per_second)
    
    elif transform.transform_type == tlib.TransformType.TRANSLATE:
        _, direction = imgui.InputFloat3("Direction", transform.direction)
        transform.direction = np.array(direction)
        if imgui.Button("Normalize"):
            transform.adjust()
        _, transform.speed = imgui.InputFloat("Speed", transform.speed)
    imgui.PopID()
        

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

        if imgui.BeginTabBar("My bar", imgui.ImGuiTabBarFlags_None):

            if imgui.BeginTabItem("Scene / Meshes", True)[0]:
                imgui.Text("")
                # Buttons for controlling mesh data in scene
                if imgui.Button("Add Mesh", default_button_size):
                    _load_mesh(state)
                if imgui.Button("Save FEM", default_button_size):
                    _save_fem(state)
                imgui.Text(f"Meshes: {len(state.meshes)}")
                if state.fem is not None:
                    imgui.Text(
                        f"FEM nodes: {state.fem.X.shape[1]}, elements: {state.fem.E.shape[1]}"
                    )
                # Per-mesh controls
                for i, m in enumerate(state.meshes):
                    if imgui.TreeNode(f"{m.name}"):
                        # Material & ICs (not yet heterogeneous in FEM build, but tracked per-mesh)
                        if imgui.TreeNode("Material"):
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
                imgui.EndTabItem()


            if imgui.BeginTabItem("Dynamics", True)[0]:
                imgui.Text("")
                _, state.aext = imgui.InputFloat3("External acceleration", state.aext)
                _, state.dt = imgui.InputFloat("Time step", state.dt)
                _, state.s = imgui.InputInt("BDF step", state.s)
                imgui.EndTabItem()

            if imgui.BeginTabItem("Constraints", True)[0]:
                imgui.Text("")
                # _, state.d_axis = imgui.InputInt("Axis (0=x,1=y,2=z)", state.d_axis)
                # _, state.d_percent = imgui.InputFloat("Percentage", state.d_percent)
                # _, state.d_extremity = imgui.InputInt(
                #     "Extremity (0=min,1=max)", state.d_extremity
                # )
                if imgui.Button("Load Transform file", default_button_size):
                    state.load_transform_file()
                if imgui.Button("Save Transform file", default_button_size):
                    state.save_transform_library()


                if state.transform_library.transforms:
                    _, state.selected_dirichlet_group = imgui.Combo(
                                "Picked Group", state.selected_dirichlet_group,
                                [t.name for t in state.transform_library.transforms]
                            )
                    
                # Read/Edit loaded transforms
                if imgui.TreeNode("Loaded Transforms"):
                    # Red buttons for buttons per transform, to distinguish from other buttons
                    imgui.PushStyleColor(imgui.ImGuiCol_Button, (0.8, 0.2, 0.2, 1.0))
                    for t in state.transform_library.transforms:
                        if imgui.TreeNode(f"{t.id}"):
                            transform_editor(t, t.id)
                            imgui.TreePop()
                    imgui.PopStyleColor(1)       
                    if (imgui.Button("Create new Transform", default_button_size) or state.editing_new_transform):
                        state.editing_new_transform = True
                        _, state.selected_ttype = imgui.Combo(
                            "Transform Type", state.selected_ttype,
                            [ttype.name for ttype in tlib.TransformType]
                        )
                        for unsaved in state.unsaved_transforms:
                            if unsaved.value == state.selected_ttype:
                                transform_editor(state.unsaved_transforms[unsaved], unsaved.value + len(state.transform_library.transforms))
                                if imgui.Button("Add Transform", default_button_size):
                                    state.add_transform(state.unsaved_transforms[unsaved])
                                    state.unsaved_transforms[unsaved] = tlib.PrimitiveTransform.make_default(unsaved)
                                    state.editing_new_transform = False
                    imgui.TreePop()
                imgui.EndTabItem()

            if imgui.BeginTabItem("Selection Boxes", True)[0]:
                imgui.Text("")
                if imgui.Button("Spawn Box Selection", default_button_size):
                    state.spawn_box_selection()
                for i, box in enumerate(state.selections):
                    
                    if imgui.TreeNode(f"{box.name}"):
                        _, target = imgui.Combo(
                            "Target", box.target.value - 1,
                            [t.name for t in box_selection.SelectionTargets]
                        )
                        box.target = box_selection.SelectionTargets(target + 1)
                        _, box.pos = imgui.SliderFloat3("Position", box.pos, -50, 50)
                        _, box.scale = imgui.SliderFloat3("Size", box.scale, 0, 10)
                        box.ps_mesh.update_vertex_positions(box.vertices * box.scale + box.pos)
                        if box.target == box_selection.SelectionTargets.CELL:
                            _, box.Y = imgui.InputFloat("Young's Modulus to Apply", box.Y)
                        if imgui.Button("Apply Box Selection", default_button_size):
                            for m in state.meshes:
                                VT = m.transformed_vertices()
                                C = m.C
                                # Get indices inside box
                                indices = box.inside_test(VT, C)
                                if box.target == box_selection.SelectionTargets.VERTEX:
                                    # Vertices only affected by Dirichlet
                                    transform = state.transform_library.transforms[state.selected_dirichlet_group]
                                    dgroup = state.dirichlet_library[transform.name].get_mesh_indices(m)
                                    for i in indices:
                                        dgroup.update_indices(i)
                                    state.dirichlet_library[transform.name].build_point_cloud()
                                elif box.target == box_selection.SelectionTargets.CELL:
                                    # Cells affected by Young's modulus change, but will expand to mass density, etc. later
                                    m.Y[indices] = box.Y
                                    m.handle.add_scalar_quantity("Young's modulus", m.Y, defined_on='cells')

                        if imgui.Button("Delete Box", default_button_size):
                            try:
                                ps.remove_surface_mesh(box.name)
                            except Exception:
                                pass
                            state.selections.pop(i)
                            imgui.TreePop()
                            break  # indices shifted
                        imgui.TreePop()
                imgui.EndTabItem()
            imgui.EndTabBar()
          
        

        # Picking IO (Setting heterogeneous constraints, eg Dirichlet)
        io = imgui.GetIO()
        if io.MouseClicked[0] and io.KeyCtrl:
          pick_result = ps.pick(screen_coords=io.MousePos)
          # print(pick_result)
          if pick_result.is_hit and pick_result.structure_type_name == "Volume Mesh" and pick_result.structure_data['element_type'] == "vertex":
            for m in state.meshes:
                if pick_result.structure_name == m.name:
                    i = pick_result.local_index
                    transform = state.transform_library.transforms[state.selected_dirichlet_group]
                    indices = state.dirichlet_library[transform.name].get_mesh_indices(m)
                    if indices is None:
                        ps.error(f"Mesh {m.name} not found in Dirichlet library for transform {group.name}.")
                        return
                    indices.update_indices(i)
                    state.dirichlet_library[transform.name].build_point_cloud()
    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
