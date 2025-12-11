# type: ignore
from pbatoolkit import pbat, pypbat
import meshio
import tkinter as tk
from tkinter import filedialog
import polyscope as ps
import polyscope.imgui as imgui
import numpy as np
import enum
import inspect
import itertools


def try_draw_tooltip(obj, name):
    if imgui.IsItemHovered():
        imgui.BeginTooltip()
        imgui.SetTooltip(getattr(type(obj), name).__doc__ or "")
        imgui.EndTooltip()


def draw_params(obj):
    for name, value in inspect.getmembers(obj):
        if (
            isinstance(getattr(type(obj), name, None), property)
            and getattr(type(obj), name).fset is None
        ):
            continue
        if isinstance(value, float):
            _, new_value = imgui.InputFloat(name, value, format="%.10f")
            try_draw_tooltip(obj, name)
            setattr(obj, name, new_value)
        elif isinstance(value, int):
            _, new_value = imgui.InputInt(name, value)
            try_draw_tooltip(obj, name)
            setattr(obj, name, new_value)
        elif isinstance(value, bool):
            _, new_value = imgui.Checkbox(name, value)
            try_draw_tooltip(obj, name)
            setattr(obj, name, new_value)
        elif isinstance(value, enum.Enum):
            enum_values = list(type(value))
            selected_idx = enum_values.index(value)
            _, selected_idx = imgui.Combo(
                name,
                selected_idx,
                [enum_value.name for enum_value in enum_values],
            )
            try_draw_tooltip(obj, name)
            setattr(obj, name, enum_values[selected_idx])


def load_dynamic_mesh():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Dynamic Mesh File",
        filetypes=[("Mesh Files", "*.mesh *.msh")],
    )
    if not file_path:
        return None, None
    mesh = meshio.read(file_path)
    V = mesh.points
    if "tetra" in mesh.cells_dict:
        T = mesh.cells_dict["tetra"]
    else:
        raise ValueError("No tetrahedral elements found in the mesh file.")
    return V, T


def load_static_mesh():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Static Mesh File",
        filetypes=[("Mesh Files", "*.obj *.off *.ply *.stl")],
    )
    if not file_path:
        return None, None
    mesh = meshio.read(file_path)
    V = mesh.points
    if "triangle" in mesh.cells_dict:
        F = mesh.cells_dict["triangle"]
    else:
        raise ValueError("No triangular faces found in the mesh file.")
    return V, F


def get_transformed_dynamic_positions(
    Vdynamic: np.ndarray,
    dynamic_mesh_vms: list[ps.VolumeMesh],
):
    Xdynamic = np.zeros((Vdynamic.shape[1], Vdynamic.shape[0]))
    node_counts = [vm.n_vertices() for vm in dynamic_mesh_vms]
    node_prefix = [0] + list(itertools.accumulate(node_counts))
    for b, vm in enumerate(dynamic_mesh_vms):
        T = vm.get_transform()
        begin, end = node_prefix[b], node_prefix[b + 1]
        Vb = Vdynamic[begin:end, :]
        VH = np.vstack([Vb.T, np.ones((1, Vb.shape[0]))])
        Xdynamic[:, begin:end] = (T @ VH)[:3, :]
    return Xdynamic.astype(dtype=np.float32)


def get_transformed_static_positions(
    Vstatic: np.ndarray,
    static_mesh_sm: ps.SurfaceMesh,
):
    T = static_mesh_sm.get_transform()
    VH = np.vstack([Vstatic.T, np.ones((1, Vstatic.shape[0]))])
    Xstatic = (T @ VH)[:3, :]
    return Xstatic.astype(dtype=np.float32)


def get_ogc_input(
    ogc_input_storage: pbat.sim.contact.ogc.InputStorage,
    Vdynamic: np.ndarray,
    dynamic_mesh_vms: list[ps.VolumeMesh],
    Vstatic: np.ndarray = None,
    static_mesh_sm: ps.SurfaceMesh = None,
):
    Xdynamic = get_transformed_dynamic_positions(Vdynamic, dynamic_mesh_vms)
    ogc_input_storage.Xdynamic = Xdynamic
    if static_mesh_sm is not None:
        Xstatic = get_transformed_static_positions(Vstatic, static_mesh_sm)
        ogc_input_storage.Xstatic = Xstatic
    return ogc_input_storage.to_input()


def main():
    Vdynamic: np.ndarray = None
    Tdynamic: np.ndarray = None
    Vstatic: np.ndarray = None
    ogc_input_storage = pbat.sim.contact.ogc.InputStorage()
    ogc_state = pbat.sim.contact.ogc.State()
    ogc_params = pbat.sim.contact.ogc.Params().with_radii(1e-3, 1e-2)
    n_dynamic_bodies = 0
    XdynamicCC = np.array([], dtype=int)
    config = pbat.geometry.DeviceConfig()
    config.verbose = 3
    device = pbat.geometry.Device(config)
    dmin = 0.0

    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Editor")
    ps.init()

    static_mesh_sm: ps.SurfaceMesh = None
    dynamic_mesh_vms: list[ps.VolumeMesh] = []

    def callback():
        nonlocal Vdynamic, Tdynamic, Vstatic, n_dynamic_bodies, XdynamicCC
        nonlocal ogc_input_storage, ogc_state, ogc_params
        nonlocal static_mesh_sm, dynamic_mesh_vms
        nonlocal dmin
        if imgui.TreeNode("I/O"):
            if imgui.Button("Load dynamic mesh"):
                try:
                    V, T = load_dynamic_mesh()
                    if V is not None and T is not None:
                        dynamic_mesh_vms.append(
                            ps.register_volume_mesh(
                                f"Dynamic Mesh {n_dynamic_bodies}", V, T
                            )
                        )
                        VCC = np.full(V.shape[0], n_dynamic_bodies)
                        XdynamicCC = (
                            VCC
                            if n_dynamic_bodies == 0
                            else np.hstack([XdynamicCC, VCC])
                        )
                        Tdynamic = (
                            T
                            if n_dynamic_bodies == 0
                            else np.vstack([Tdynamic, T + Vdynamic.shape[0]])
                        )
                        Vdynamic = (
                            V if n_dynamic_bodies == 0 else np.vstack([Vdynamic, V])
                        )
                        n_dynamic_bodies += 1
                        ogc_input_storage.dynamic_mesh.construct_from_tetrahedral_mesh(
                            Tdynamic.T, XdynamicCC, n_dynamic_bodies
                        )
                except Exception as e:
                    ps.error(f"Failed to load dynamic mesh: {e}")
            if imgui.Button("Load static mesh"):
                try:
                    V, F = load_static_mesh()
                    if V is not None and F is not None:
                        Vstatic = V
                        ogc_input_storage.static_mesh.construct_from_triangle_mesh(
                            F.T, np.zeros(V.shape[0], dtype=np.int64)
                        )
                        static_mesh_sm = ps.register_surface_mesh("Static Mesh", V, F)
                except Exception as e:
                    ps.error(f"Failed to load static mesh: {e}")
            imgui.TreePop()
        if imgui.TreeNode("OGC"):
            if imgui.Button("Initialize"):
                try:
                    ogc_input = get_ogc_input(
                        ogc_input_storage,
                        Vdynamic,
                        dynamic_mesh_vms,
                        Vstatic,
                        static_mesh_sm,
                    )
                    ogc_state.initialize(device, ogc_input, ogc_params)
                except Exception as e:
                    ps.error(f"Failed to initialize OGC: {e}")
            if imgui.Button("Execute"):
                try:
                    ogc_input = get_ogc_input(
                        ogc_input_storage,
                        Vdynamic,
                        dynamic_mesh_vms,
                        Vstatic,
                        static_mesh_sm,
                    )
                    ogc_state.prepare_for_execution(ogc_input, ogc_params)
                    pbat.sim.contact.ogc.vertex_facet_contact_detection(
                        ogc_input, ogc_params, ogc_state
                    )
                    pbat.sim.contact.ogc.edge_edge_contact_detection(
                        ogc_input, ogc_params, ogc_state
                    )
                    pbat.sim.contact.ogc.update_displacement_bounds(
                        ogc_input, ogc_params, ogc_state
                    )
                    dmin = ogc_state.bv.min()
                except Exception as e:
                    ps.error(f"Failed to execute OGC: {e}")
            imgui.Text(f"Minimum displacement: {dmin:.4f}")
            imgui.TreePop()
        if imgui.TreeNode("Params"):
            draw_params(ogc_params)
            imgui.TreePop()

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
