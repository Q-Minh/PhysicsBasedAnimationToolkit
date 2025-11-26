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
from ..utils.pick import BoxSelection, SelectionTargets
from ..utils.transform_library import TransformLibrary


class BoxSelectionList:
    _default_value: typing.Any
    _selection_target: SelectionTargets
    _listener: typing.Callable[[int, typing.Any, np.ndarray[int]], None]
    _recycled_indices: list[int]
    selectors: list[BoxSelection]

    def __init__(self, selection_target: SelectionTargets, default_value, listener):
        self._selection_target = selection_target
        self._default_value = default_value
        self._listener = listener
        self._recycled_indices = []
        self.selectors = []

    def on_selector_added(self, prop_type: str):
        self.selectors.append(
            BoxSelection(
                f"{prop_type} - {self._get_new_id()}",
                prop_type,
                self._default_value,
                self._selection_target,
                self._listener,
            )
        )
        self.selectors[-1].on_added()

    def on_selector_removed(self, idx: int):
        selector = self.selectors.pop(idx)
        selector.on_removed()
        id = int(selector.name.split(" - ")[-1])
        self._recycled_indices.append(id)

    def _get_new_id(self) -> int:
        if self._recycled_indices:
            return self._recycled_indices.pop()
        else:
            return len(self.selectors)


class Scene:
    _tet_elastic_bodies: list[TetrahedralElastodynamicsBody]
    _recycled_tet_elastic_body_indices: list[int]
    _selector_lists: dict[str, BoxSelectionList]
    _transform_library: TransformLibrary

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
                lambda b, prop_value, inds: self._tet_elastic_bodies[
                    b
                ].set_dirichlet_group(prop_value, inds),
            ),
        }
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
                        if imgui.Button("Delete", default_button_size):
                            self._remove_tet_elastic_body(b)
                        imgui.TreePop()
                    imgui.PopID()
                imgui.EndTabItem()

            if imgui.BeginTabItem("Transforms", True, tab_flags)[0]:
                self._transform_library.draw()
                imgui.EndTabItem()

            if imgui.BeginTabItem("Selection", True, tab_flags)[0]:
                for prop_name, box_selection_list in self._selector_lists.items():
                    if imgui.TreeNode(prop_name):
                        if imgui.Button("Add", default_button_size):
                            box_selection_list.on_selector_added(prop_name)
                        for s, selector in enumerate(box_selection_list.selectors):
                            imgui.PushID(f"{prop_name} - {s}")
                            if imgui.TreeNode(selector.name):
                                selector.draw(self._tet_elastic_bodies)
                                imgui.TreePop()
                            imgui.PopID()
                        imgui.TreePop()
                imgui.EndTabItem()

            if imgui.BeginTabItem("Session", True, tab_flags)[0]:
                if imgui.Button("Load session", default_button_size):
                    self._load_session()
                if imgui.Button("Save session", default_button_size):
                    self._save_session()
                imgui.EndTabItem()
            imgui.EndTabBar()

        # No matter what tab we're in, undirty any dirty bodies at the end of the frame
        for b, body in enumerate(self._tet_elastic_bodies):
            if body.dirty:
                body.undirty(n_dirichlet_groups=len(self._transform_library.transforms))

    def _save_session(self):
        # TODO: Implement saving session
        ps.warning("Save session not implemented yet.")

    def _load_session(self):
        # TODO: Implement loading session
        ps.warning("Load session not implemented yet.")

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
                self._tet_elastic_bodies.append(body)
            except Exception as e:
                ps.error(f"Error loading tetrahedral mesh:\n{e}")
            finally:
                root.destroy()

    def _remove_tet_elastic_body(self, b: int):
        body = self._tet_elastic_bodies.pop(b)
        idx = int(body.name.split(" - ")[-1])
        body.on_mesh_removed()
        self._recycled_tet_elastic_body_indices.append(idx)

    def _load_transform_library(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select transform library file",
            defaultextension=".h5",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
        )
        if file_path:
            try:
                self._transform_library.deserialize(file_path)
            except Exception as e:
                ps.error(f"Error loading transform library:\n{e}")
            finally:
                root.destroy()
