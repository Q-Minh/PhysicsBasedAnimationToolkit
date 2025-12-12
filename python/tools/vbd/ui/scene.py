# type: ignore
from .tetrahedral_elastodynamics_body import TetrahedralElastodynamicsBody
import polyscope as ps
import polyscope.imgui as imgui
import tkinter as tk
from tkinter import filedialog
import meshio
import os
import typing
import numpy as np
import h5py as h5
from .utils.box_selection import BoxSelection, SelectionTargets
from .utils.transform_library import TransformLibrary
from . import styles


class BoxSelectionList:
    _default_value: typing.Any
    _selection_target: SelectionTargets
    _listener: typing.Callable[[int, typing.Any, np.ndarray[int]], None]
    _recycled_indices: list[int]
    _selectors: list[BoxSelection]

    def __init__(self, selection_target: SelectionTargets, default_value, listener):
        self._selection_target = selection_target
        self._default_value = default_value
        self._listener = listener
        self._recycled_indices = []
        self._selectors = []

    def on_selector_added(self, prop_type: str):
        self._selectors.append(
            BoxSelection(
                f"{prop_type} - {self._get_new_id()}",
                prop_type,
                self._default_value,
                self._selection_target,
                self._listener,
            )
        )
        self._selectors[-1].on_added()

    def remove_selector(self, idx: int):
        selector = self._selectors.pop(idx)
        selector.on_removed()
        id = int(selector.name.split(" - ")[-1])
        self._recycled_indices.append(id)

    def _get_new_id(self) -> int:
        if self._recycled_indices:
            return self._recycled_indices.pop()
        else:
            return len(self._selectors)


class Scene:
    _tet_elastic_bodies: list[TetrahedralElastodynamicsBody]
    _recycled_tet_elastic_body_indices: list[int]
    _transform_library: TransformLibrary
    _selector_lists: dict[str, BoxSelectionList]
    _current_selection_property_idx: int

    def __init__(self):
        self._tet_elastic_bodies = []
        self._recycled_tet_elastic_body_indices = []
        self._selector_lists = {
            "Young's Modulus": BoxSelectionList(
                SelectionTargets.CELL,
                1e6,
                lambda b, prop_value, inds: self._tet_elastic_bodies[
                    b
                ].set_young_modulus(prop_value, inds),
            ),
            "Poisson's Ratio": BoxSelectionList(
                SelectionTargets.CELL,
                0.45,
                lambda b, prop_value, inds: self._tet_elastic_bodies[
                    b
                ].set_poisson_ratio(prop_value, inds),
            ),
            "Mass Density": BoxSelectionList(
                SelectionTargets.CELL,
                1e3,
                lambda b, prop_value, inds: self._tet_elastic_bodies[
                    b
                ].set_mass_density(prop_value, inds),
            ),
            "External Load": BoxSelectionList(
                SelectionTargets.CELL,
                np.zeros(3),
                lambda b, prop_value, inds: self._tet_elastic_bodies[
                    b
                ].set_external_load(prop_value, inds),
            ),
            "Initial Velocity": BoxSelectionList(
                SelectionTargets.VERTEX,
                np.zeros(3),
                lambda b, prop_value, inds: self._tet_elastic_bodies[
                    b
                ].set_initial_velocities(prop_value, inds),
            ),
            "Dirichlet Group": BoxSelectionList(
                SelectionTargets.VERTEX,
                1,
                lambda b, prop_value, inds: self._on_dirichlet_group_applied(
                    b, prop_value, inds
                ),
            ),
        }
        self._current_selection_property_idx = 0
        self._transform_library = TransformLibrary()

    def draw(self):
        default_button_size = [imgui.GetWindowWidth() / 2.1, 0]
        tab_flags = (
            imgui.ImGuiTabBarFlags_Reorderable
            | imgui.ImGuiTabBarFlags_FittingPolicyScroll
            | imgui.ImGuiTabBarFlags_TabListPopupButton
        )
        if imgui.BeginTabBar("Mode bar", tab_flags):
            if imgui.BeginTabItem("Objects", True, tab_flags)[0]:
                if imgui.Button("Add Tetrahedral Body", default_button_size):
                    self._load_tet_elastic_body()
                for b, body in enumerate(self._tet_elastic_bodies):
                    imgui.PushID(body.name)
                    if imgui.TreeNode(body.name):
                        body.draw()
                        if imgui.TreeNode("Delete"):
                            styles.set_style_danger()
                            if imgui.Button("Delete", default_button_size):
                                self._remove_tet_elastic_body(b)
                            styles.pop_most_recent_style()
                            imgui.TreePop()
                        imgui.TreePop()
                    imgui.PopID()
                imgui.EndTabItem()

            if imgui.BeginTabItem("Transforms", True, tab_flags)[0]:
                self._transform_library.draw()
                imgui.EndTabItem()

            if imgui.BeginTabItem("Selection", True, tab_flags)[0]:
                prop_names = list(self._selector_lists.keys())
                _, self._current_selection_property_idx = imgui.Combo(
                    "Property",
                    self._current_selection_property_idx,
                    prop_names,
                )
                prop_name = prop_names[self._current_selection_property_idx]
                box_selection_list = self._selector_lists[prop_name]
                if imgui.Button("Add", default_button_size):
                    box_selection_list.on_selector_added(prop_name)
                for s, selector in enumerate(box_selection_list._selectors):
                    imgui.PushID(f"{prop_name} - {s}")
                    if imgui.TreeNode(selector.name):
                        selector.draw(self._tet_elastic_bodies)
                        imgui.SameLine()
                        if imgui.TreeNode("Delete"):
                            styles.set_style_danger()
                            if imgui.Button("Delete", default_button_size):
                                box_selection_list.remove_selector(s)
                            styles.pop_most_recent_style()
                            imgui.TreePop()
                        imgui.TreePop()
                    imgui.PopID()
                imgui.EndTabItem()

            if imgui.BeginTabItem("Session", True, tab_flags)[0]:
                if imgui.Button("Load session", default_button_size):
                    self._load_session()
                if imgui.Button("Load session (bodies only)", default_button_size):
                    self._load_session(bodies_only=True)
                if imgui.Button("Save session", default_button_size):
                    self._save_session()
                imgui.EndTabItem()
            imgui.EndTabBar()

        is_any_body_dirty = False
        for b, body in enumerate(self._tet_elastic_bodies):
            if body.dirty:
                body.undirty(n_dirichlet_groups=len(self._transform_library.transforms))
                is_any_body_dirty = True
        mesh_names = [body.name for body in self._tet_elastic_bodies]
        mesh_verts = [body.VT for body in self._tet_elastic_bodies]
        for transform in self._transform_library.transforms:
            if transform.dirty or is_any_body_dirty:
                transform.undirty(mesh_names=mesh_names, mesh_verts=mesh_verts)

    def set_visible(self, visible: bool):
        for body in self._tet_elastic_bodies:
            body.set_visible(visible)
        for prop_name, box_selection_list in self._selector_lists.items():
            for selector in box_selection_list._selectors:
                selector.set_visible(visible)
        for transform in self._transform_library.transforms:
            transform.set_visible(visible)

    @property
    def tet_elastic_bodies(self) -> list[TetrahedralElastodynamicsBody]:
        return self._tet_elastic_bodies

    @property
    def transform_library(self) -> TransformLibrary:
        return self._transform_library

    def _save_session(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(
            title="Save session (HDF5)",
            defaultextension=".h5",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
        )
        try:
            if file_path:
                with h5.File(file_path, "w") as f:
                    self._serialize_fem_tet_elastic_bodies(
                        f.create_group("fem_tet_elastic_bodies")
                    )
                    self._transform_library.serialize(
                        f.create_group("transform_library")
                    )
                    f["recycled_tet_elastic_body_indices"] = np.array(
                        self._recycled_tet_elastic_body_indices, dtype=int
                    )
        except Exception as e:
            ps.error(f"Error saving session:\n{e}")
        finally:
            root.destroy()

    def _load_session(self, bodies_only=False):
        body_info = "- Without transforms" if bodies_only else ""
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Load session (HDF5)" + body_info,
            defaultextension=".h5",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
        )
        try:
            if file_path:
                with h5.File(file_path, "r") as f:
                    self._teardown(bodies_only=bodies_only)
                    self._buildup(f, bodies_only=bodies_only)

        except Exception as e:
            ps.error(f"Error loading session:\n{e}")
        finally:
            root.destroy()

    def _get_new_id(self) -> int:
        if self._recycled_tet_elastic_body_indices:
            return self._recycled_tet_elastic_body_indices.pop()
        else:
            return len(self._tet_elastic_bodies)

    def _load_tet_elastic_body(self):
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
        if file_path:
            try:
                imesh = meshio.read(file_path)
                V = imesh.points
                T = imesh.cells_dict["tetra"]
                filename = os.path.basename(file_path)
                id = self._get_new_id()
                body = TetrahedralElastodynamicsBody()
                body.on_mesh_loaded(f"{filename} - {id}", V, T)
                self._transform_library.on_mesh_added(body.name)
                self._tet_elastic_bodies.append(body)
            except Exception as e:
                ps.error(f"Error loading tetrahedral mesh:\n{e}")
            finally:
                root.destroy()

    def _remove_tet_elastic_body(self, b: int):
        body = self._tet_elastic_bodies.pop(b)
        idx = int(body.name.split(" - ")[-1])
        self._transform_library.on_mesh_removed(body.name)
        body.on_mesh_removed()
        self._recycled_tet_elastic_body_indices.append(idx)

    def _on_dirichlet_group_applied(self, b: int, group: int, inds: np.ndarray[int]):
        body = self._tet_elastic_bodies[b]
        is_dirichlet_constraint_removal = group == 0
        if is_dirichlet_constraint_removal:
            self._transform_library.on_dirichlet_nodes_removed(body.name, inds)
        else:
            self._transform_library.on_dirichlet_nodes_added(group, body.name, inds)

    def _serialize_fem_tet_elastic_bodies(self, grp: h5.Group):
        grp.attrs["num_tet_elastic_bodies"] = len(self._tet_elastic_bodies)
        for b, body in enumerate(self._tet_elastic_bodies):
            body_grp = grp.create_group(f"{b}")
            body.serialize(body_grp)

    def _deserialize_fem_tet_elastic_bodies(self, grp: h5.Group):
        num_bodies = grp.attrs["num_tet_elastic_bodies"]
        for b in range(num_bodies):
            body_grp = grp[f"{b}"]
            body = TetrahedralElastodynamicsBody()
            body.deserialize(body_grp)
            self._tet_elastic_bodies.append(body)

    def _teardown(self, bodies_only=False):
        if not bodies_only:
            self._transform_library.empty()
        else:
            for body in self._tet_elastic_bodies:
                self._transform_library.on_mesh_removed(body.name)

        for body in self._tet_elastic_bodies:
            body.on_mesh_removed()
        self._tet_elastic_bodies = []
        self._recycled_tet_elastic_body_indices = []

    def _buildup(self, f, bodies_only=False):
        self._deserialize_fem_tet_elastic_bodies(f["fem_tet_elastic_bodies"])
        self._recycled_tet_elastic_body_indices = list(
            f["recycled_tet_elastic_body_indices"][:]
        )
        if not bodies_only:
            self._transform_library.deserialize(f["transform_library"])
        else:
            for body in self._tet_elastic_bodies:
                self._transform_library.on_mesh_added(body.name)
