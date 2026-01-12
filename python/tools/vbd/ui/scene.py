# type: ignore
from .tetrahedral_elastodynamics_body import TetrahedralElastodynamicsBody
from .static_mesh_collider import StaticMeshCollider
import polyscope as ps
import polyscope.imgui as imgui
import tkinter as tk
from tkinter import filedialog
import meshio
import os
import typing
import numpy as np
import h5py as h5
from .utils.box_selection import Selection, BoxSelection, RegionSelection, SelectionTargets
from .utils.transform_library import TransformLibrary
from .utils import styles


class SelectionList:
    _default_value: typing.Any
    _selection_target: SelectionTargets
    _listener: typing.Callable[[int, typing.Any, np.ndarray[int]], None]
    _recycled_indices: list[int]
    _selectors: list[Selection]
    _list_type: str

    def __init__(self, selection_target: SelectionTargets, default_value, list_type: str, listener):
        self._selection_target = selection_target
        self._default_value = default_value
        self._listener = listener
        self._recycled_indices = []
        self._selectors = []
        self._list_type = list_type
        
    def on_selector_added(self, prop_type: str):
        if self._list_type == BoxSelection.__name__:
            self._selectors.append(
                BoxSelection(
                    f"{prop_type} - {self._get_new_id()}",
                    prop_type,
                    self._default_value,
                    self._selection_target,
                    self._listener,
                )
            )
        elif self._list_type == RegionSelection.__name__:
            self._selectors.append(
                RegionSelection(
                    f"{prop_type} - {self._get_new_id()}",
                    prop_type,
                    self._default_value,
                    self._listener,
                )
            )
        else:
            raise ValueError(f"No SelectionList support for type {self._list_type}")
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
    _box_selector_lists: dict[str, SelectionList]
    _region_selector_lists: dict[str, SelectionList]
    _current_selection_property_idx: int
    _static_mesh_colliders: list[StaticMeshCollider]

    def set_young_modulus(self, b, prop_value, inds): 
        self._tet_elastic_bodies[b].set_young_modulus(prop_value, inds)

    def set_poisson_ratio(self, b, prop_value, inds): 
        self._tet_elastic_bodies[b].set_poisson_ratio(prop_value, inds)
    
    def set_mass_density(self, b, prop_value, inds): 
        self._tet_elastic_bodies[b].set_mass_density(prop_value, inds)

    def set_external_load(self, b, prop_value, inds): 
        self._tet_elastic_bodies[b].set_external_load(prop_value, inds)

    def set_initial_velocities(self, b, prop_value, inds): 
        self._tet_elastic_bodies[b].set_initial_velocities(prop_value, inds),


    def __init__(self):
        self._tet_elastic_bodies = []
        self._recycled_tet_elastic_body_indices = []

        name_to_vars = {
            "Young's Modulus": {
                "default": 1e6,
                "func": self.set_young_modulus
            },
            "Poisson's Ratio": {
                "default": 0.45, 
                "func": self.set_poisson_ratio
            }, 
            "Mass Density": {
                "default": 1e3,
                "func": self.set_mass_density
            },
            "External Load": {
                "default": np.zeros(3),
                "func": self.set_external_load
            }
        }
        self._box_selector_lists = {}
        self._region_selector_lists = {}

        for (name, item) in name_to_vars.items():
            self._box_selector_lists[name] = SelectionList(
                SelectionTargets.CELL,
                item["default"],
                BoxSelection.__name__,
                item["func"]
            )
            self._region_selector_lists[name] = SelectionList(
                SelectionTargets.CELL,
                item["default"],
                RegionSelection.__name__,
                item["func"]
            )

        self._box_selector_lists["Initial Velocity"] = SelectionList(
            SelectionTargets.VERTEX,
            np.zeros(3),
            BoxSelection.__name__,
            self.set_initial_velocities
        )
        self._box_selector_lists["Dirichlet Group"] = SelectionList(
            SelectionTargets.VERTEX,
            1,
            BoxSelection.__name__,
            lambda b, prop_value, inds: self._on_dirichlet_group_applied(
                b, prop_value, inds
            ),
        )
        

        self._current_selection_property_idx = 0
        self._transform_library = TransformLibrary()
        self._static_mesh_colliders = []

    def draw(self):
        tab_flags = styles.default_tab_flags()
        if imgui.BeginTabBar("Mode bar", tab_flags):
            if imgui.BeginTabItem("Objects", True, tab_flags)[0]:
                if imgui.Button(
                    "Add Tetrahedral Body", styles.default_button_size()
                ):
                    self._load_tet_elastic_body()
                if imgui.Button(
                    "Add Static Mesh Collider", styles.default_button_size()
                ):
                    self._load_static_mesh_collider()
                if imgui.TreeNode("Tetrahedral Elastic Bodies"):
                    for b, body in enumerate(self._tet_elastic_bodies):
                        imgui.PushID(body.name)
                        if imgui.TreeNode(body.name):
                            body.draw()
                            styles.set_style_danger()
                            if imgui.Button(
                                styles.delete_key(), styles.small_button_size()
                            ):
                                body = self._tet_elastic_bodies.pop(b)
                                body.on_mesh_removed()
                            styles.pop_most_recent_style()
                            imgui.TreePop()
                        imgui.PopID()
                    imgui.TreePop()
                if imgui.TreeNode("Static Mesh Colliders"):
                    self._draw_static_mesh_colliders()
                    imgui.TreePop()
                imgui.EndTabItem()

            if imgui.BeginTabItem("Transforms", True, tab_flags)[0]:
                self._transform_library.draw()
                imgui.EndTabItem()

            if imgui.BeginTabItem("Selection", True, tab_flags)[0]:
                prop_names = list(self._box_selector_lists.keys())
                _, self._current_selection_property_idx = imgui.Combo(
                    "Property",
                    self._current_selection_property_idx,
                    prop_names,
                )
                prop_name = prop_names[self._current_selection_property_idx]
                if imgui.BeginTabBar("Selection_types", tab_flags):
                    if imgui.BeginTabItem("Box", True, tab_flags)[0]:
                        box_selection_list = self._box_selector_lists[prop_name]
                        if imgui.Button("Add", styles.default_button_size()):
                            box_selection_list.on_selector_added(prop_name)
                        for s, selector in enumerate(box_selection_list._selectors):
                            imgui.PushID(f"{prop_name} - {s}")
                            if imgui.TreeNode(selector.name):
                                selector.draw(self._tet_elastic_bodies)
                                imgui.SameLine()
                                styles.set_style_danger()
                                if imgui.Button(
                                    styles.delete_key(), styles.small_button_size()
                                ):
                                    box_selection_list.remove_selector(s)
                                styles.pop_most_recent_style()
                                imgui.TreePop()
                            imgui.PopID()
                        imgui.EndTabItem()

                    if imgui.BeginTabItem("Region", True, tab_flags)[0]:
                        region_selection_list = self._region_selector_lists.get(prop_name)
                        if region_selection_list is None:
                            imgui.Text(f"Region Selection not available for {prop_name}")
                        else:
                            if imgui.Button("Add", styles.default_button_size()):
                                region_selection_list.on_selector_added(prop_name)
                            for s, selector in enumerate(region_selection_list._selectors):
                                imgui.PushID(f"{prop_name} - {s}")
                                if imgui.TreeNode(selector.name):
                                    selector.draw(self._tet_elastic_bodies)
                                    imgui.SameLine()
                                    styles.set_style_danger()
                                    if imgui.Button(
                                        styles.delete_key(), styles.small_button_size()
                                    ):
                                        region_selection_list.remove_selector(s)
                                    styles.pop_most_recent_style()
                                    imgui.TreePop()
                                imgui.PopID()
                        imgui.EndTabItem()
                    imgui.EndTabBar()
                imgui.EndTabItem()

            if imgui.BeginTabItem("Session", True, tab_flags)[0]:
                if imgui.Button("Load session", styles.default_button_size()):
                    self._load_session()
                if imgui.Button(
                    "Load session (bodies only)", styles.default_button_size()
                ):
                    self._load_session(bodies_only=True)
                if imgui.Button("Save session", styles.default_button_size()):
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
        for smc in self._static_mesh_colliders:
            smc.set_visible(visible)
        for prop_name, box_selection_list in self._box_selector_lists.items():
            for selector in box_selection_list._selectors:
                selector.set_visible(visible)
        for transform in self._transform_library.transforms:
            transform.set_visible(visible)

    @property
    def tet_elastic_bodies(self) -> list[TetrahedralElastodynamicsBody]:
        return self._tet_elastic_bodies

    @property
    def static_mesh_colliders(self) -> list[StaticMeshCollider]:
        return self._static_mesh_colliders

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
                    self._serialize_static_mesh_colliders(
                        f.create_group("static_mesh_colliders")
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
            #try:
            imesh = meshio.read(file_path)
            V = imesh.points
            T = imesh.cells_dict["tetra"]
            R = imesh.cell_data["medit:ref"][0].T
            filename = os.path.basename(file_path)
            id = self._get_new_id()
            body = TetrahedralElastodynamicsBody()
            body.on_mesh_loaded(f"{filename} - {id}", V, T, R)
            self._transform_library.on_mesh_added(body.name)
            self._tet_elastic_bodies.append(body)
            # except Exception as e:
            #     ps.error(f"Error loading tetrahedral mesh:\n{e}")
            # finally:
            #     root.destroy()

    def _load_static_mesh_collider(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select static mesh collider file",
            defaultextension=".obj",
            filetypes=[
                ("Mesh files", "*.obj *.stl *.ply"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            try:
                imesh = meshio.read(file_path)
                V = imesh.points
                if "triangle" not in imesh.cells_dict:
                    raise ValueError("Mesh must contain triangle faces.")
                F = imesh.cells_dict["triangle"]
                filename = os.path.basename(file_path)
                identifier = f"{filename} - {len(self._static_mesh_colliders)}"
                smc = StaticMeshCollider()
                smc.on_mesh_added(identifier, V, F)
                self._static_mesh_colliders.append(smc)
            except Exception as e:
                ps.error(f"Error loading static mesh collider:\n{e}")
            finally:
                root.destroy()

    def _draw_static_mesh_colliders(self):
        for idx, smc in enumerate(self._static_mesh_colliders):
            if imgui.TreeNode(smc.name):
                smc.draw()
                styles.set_style_danger()
                if imgui.Button(
                    styles.delete_key(), styles.small_button_size()
                ):
                    smc = self._static_mesh_colliders.pop(idx)
                    smc.on_mesh_removed()
                styles.pop_most_recent_style()
                imgui.TreePop()

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

    def _serialize_static_mesh_colliders(self, grp: h5.Group):
        grp.attrs["num_static_mesh_colliders"] = len(self._static_mesh_colliders)
        for i, smc in enumerate(self._static_mesh_colliders):
            collider_grp = grp.create_group(f"{i}")
            smc.serialize(collider_grp)

    def _deserialize_fem_tet_elastic_bodies(self, grp: h5.Group):
        num_bodies = grp.attrs["num_tet_elastic_bodies"]
        for b in range(num_bodies):
            body_grp = grp[f"{b}"]
            body = TetrahedralElastodynamicsBody()
            body.deserialize(body_grp)
            self._tet_elastic_bodies.append(body)

    def _deserialize_static_mesh_colliders(self, grp: h5.Group):
        num_colliders = grp.attrs["num_static_mesh_colliders"]
        for i in range(num_colliders):
            collider_grp = grp[f"{i}"]
            smc = StaticMeshCollider()
            smc.deserialize(collider_grp)
            self._static_mesh_colliders.append(smc)

    def _teardown(self, bodies_only=False):
        if not bodies_only:
            self._transform_library.empty()
            for smc in self._static_mesh_colliders:
                smc.on_mesh_removed()
            self._static_mesh_colliders = []
        else:
            for body in self._tet_elastic_bodies:
                self._transform_library.on_mesh_removed(body.name)

        for body in self._tet_elastic_bodies:
            body.on_mesh_removed()
        self._tet_elastic_bodies = []
        self._recycled_tet_elastic_body_indices = []

    def _buildup(self, f, bodies_only=False):
        self._deserialize_fem_tet_elastic_bodies(f["fem_tet_elastic_bodies"])
        if "static_mesh_colliders" in f:
            self._deserialize_static_mesh_colliders(f["static_mesh_colliders"])
        self._recycled_tet_elastic_body_indices = list(
            f["recycled_tet_elastic_body_indices"][:]
        )
        if not bodies_only:
            self._transform_library.deserialize(f["transform_library"])
        else:
            for body in self._tet_elastic_bodies:
                self._transform_library.on_mesh_added(body.name)
