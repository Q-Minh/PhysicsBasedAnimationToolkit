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
from .utils.box_selection import Selection, BoxSelection, RegionSelection, CylinderSelection, SelectionTargets
from .utils.transform_library import TransformLibrary
from .utils import styles


class SelectorList:
    """List of selection objects, per property"""
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
        """Create a new selector"""
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
        elif self._list_type == CylinderSelection.__name__:
            self._selectors.append(
                CylinderSelection(
                    f"{prop_type} - {self._get_new_id()}",
                    prop_type,
                    self._default_value,
                    self._selection_target,
                    self._listener,
                )
            )
        else:
            raise ValueError(f"No SelectorList support for type {self._list_type}")
        self._selectors[-1].on_added()

    def remove_selector(self, idx: int):
        """Remove a selector"""
        selector = self._selectors.pop(idx)
        selector.on_removed()
        id = int(selector.name.split(" - ")[-1])
        self._recycled_indices.append(id)

    def _get_new_id(self) -> int:
        if self._recycled_indices:
            return self._recycled_indices.pop()
        else:
            return len(self._selectors)

class SelectorListManager:
    """Class that will monitor a selector list"""
    selector_list: dict[str, SelectorList]
    usable_on: list[SelectionTargets]
    class_name: str

    def __init__(self, selector_list: dict[str, SelectorList], usable_on: list[SelectionTargets], class_name: str):
        self.selector_list = selector_list
        self.usable_on = usable_on
        self.class_name = class_name

    def is_usable_on(self, target: SelectionTargets) -> bool:
        return target in self.usable_on

class Scene:
    _tet_elastic_bodies: list[TetrahedralElastodynamicsBody]
    _recycled_tet_elastic_body_indices: list[int]
    _transform_library: TransformLibrary
    _pattern_data: str = ""

    _all_selectors: dict[str, SelectorListManager]
    # _box_selector_lists: dict[str, SelectorList]
    # _region_selector_lists: dict[str, SelectorList]
    # _cylinder_selector_lists: dict[str, SelectorList]
    _current_selection_property_idx: int
    _static_mesh_colliders: list[StaticMeshCollider]
    _selection_props: dict[str, dict[str, typing.Any]]

    def set_young_modulus(self, b, prop_value, inds): 
        self._tet_elastic_bodies[b].set_young_modulus(prop_value, inds)

    def set_poisson_ratio(self, b, prop_value, inds): 
        self._tet_elastic_bodies[b].set_poisson_ratio(prop_value, inds)
    
    def set_mass_density(self, b, prop_value, inds): 
        self._tet_elastic_bodies[b].set_mass_density(prop_value, inds)

    def set_external_load(self, b, prop_value, inds): 
        self._tet_elastic_bodies[b].set_external_load(prop_value, inds)

    def set_initial_velocities(self, b, prop_value, inds): 
        self._tet_elastic_bodies[b].set_initial_velocities(prop_value, inds)

    def set_dirichlet_group(self, b, prop_value, inds):
        self._on_dirichlet_group_applied(b, prop_value, inds)


    def __init__(self):
        self._tet_elastic_bodies = []
        self._recycled_tet_elastic_body_indices = []

        self._selection_props = {
            "Young's Modulus": {
                "default": 1e6,
                "func": self.set_young_modulus,
                "target": SelectionTargets.CELL
            },
            "Poisson's Ratio": {
                "default": 0.45, 
                "func": self.set_poisson_ratio,
                "target": SelectionTargets.CELL
            }, 
            "Mass Density": {
                "default": 1e3,
                "func": self.set_mass_density,
                "target": SelectionTargets.CELL
            },
            "External Load": {
                "default": np.zeros(3),
                "func": self.set_external_load,
                "target": SelectionTargets.CELL
            },
            "Initial Velocity": {
                "default": np.zeros(3),
                "func": self.set_initial_velocities,
                "target": SelectionTargets.VERTEX
            },
            "Dirichlet Group": {
                "default": 1,
                "func": self.set_dirichlet_group,
                "target": SelectionTargets.VERTEX
            }
        }

        self._all_selectors = {
            "Box": SelectorListManager({}, [SelectionTargets.CELL, SelectionTargets.VERTEX], BoxSelection.__name__),
            "Region": SelectorListManager({}, [SelectionTargets.CELL], RegionSelection.__name__),
            "Cylinder": SelectorListManager({}, [SelectionTargets.CELL, SelectionTargets.VERTEX], CylinderSelection.__name__),
        }

        for (_, manager) in self._all_selectors.items():
            for (name, item) in self._selection_props.items():
                if item["target"] in manager.usable_on:
                    manager.selector_list[name] = SelectorList(
                        item["target"],
                        item["default"],
                        manager.class_name,
                        item["func"]
                    )

        self._current_selection_property_idx = 0
        self._tet_body_to_dup_idx = 0
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
                if imgui.Button(
                    "Load Pattern File", styles.default_button_size()
                ):
                    self._load_pattern_file()
                imgui.SameLine()
                imgui.Text(self._pattern_data)
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
                if imgui.TreeNode("Pattern Data"):
                    if self._pattern_data:
                        imgui.Text(self._pattern_data)

                    _, self._tet_body_to_dup_idx = imgui.Combo(
                        "Apply to",
                        self._tet_body_to_dup_idx,
                        [body.name for body in self._tet_elastic_bodies],
                    )
                    body = self._tet_elastic_bodies[self._tet_body_to_dup_idx]
                    if imgui.Button("Apply pattern data to selected body", styles.default_button_size()):
                        self.duplicate_tet_elastic_body(body, self._pattern_data)
                    imgui.TreePop()
                imgui.EndTabItem()

            if imgui.BeginTabItem("Transforms", True, tab_flags)[0]:
                self._transform_library.draw()
                imgui.EndTabItem()

            if imgui.BeginTabItem("Selection", True, tab_flags)[0]:
                prop_names = list(self._selection_props.keys())
                _, self._current_selection_property_idx = imgui.Combo(
                    "Property",
                    self._current_selection_property_idx,
                    prop_names,
                )
                prop_name = prop_names[self._current_selection_property_idx]
                if imgui.BeginTabBar("Selection_types", tab_flags):
                    for (selector_name, manager) in self._all_selectors.items():
                        imgui.PushID(selector_name)
                        if imgui.BeginTabItem(selector_name, True, tab_flags)[0]:
                            if not manager.is_usable_on(self._selection_props[prop_name]["target"]):
                                imgui.Text(f"{selector_name} Selection not available for {prop_name}")
                            
                            else:
                                selection_list = manager.selector_list.get(prop_name)
                                if imgui.Button("Add", styles.default_button_size()):
                                    selection_list.on_selector_added(prop_name)
                                for s, selector in enumerate(selection_list._selectors):
                                    imgui.PushID(f"{prop_name} - {s}")
                                    if imgui.TreeNode(selector.name):
                                        selector.draw(self._tet_elastic_bodies)
                                        imgui.SameLine()
                                        styles.set_style_danger()
                                        if imgui.Button(
                                            styles.delete_key(), styles.small_button_size()
                                        ):
                                            selection_list.remove_selector(s)
                                        styles.pop_most_recent_style()
                                        imgui.TreePop()
                                    imgui.PopID()
                            imgui.EndTabItem()
                        imgui.PopID()
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

        for selector_name, manager in self._all_selectors.items():
            if selector_name in ["Box", "Cylinder"]:
                for _, selector_list in manager.selector_list.items():
                    for selector in selector_list._selectors:
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
            def read_mesh(offset=(0, 0, 0)):
                imesh = meshio.read(file_path)
                V = imesh.points + np.array(offset)
                T = imesh.cells_dict["tetra"]
                R = imesh.cell_data["medit:ref"][0].T if "medit:ref" in imesh.cell_data else None
                filename = os.path.basename(file_path)
                id = self._get_new_id()
                body = TetrahedralElastodynamicsBody()
                body.on_mesh_loaded(f"{filename} - {id}", V, T, R)
                self._transform_library.on_mesh_added(body.name)
                self._tet_elastic_bodies.append(body)

            try:
                if self._pattern_data:
                    with open(self._pattern_data, "r") as f:
                        # this file is a csv with x, y, z coordinates. 
                        # for each row, we want to load a mesh using read_mesh, offset using the x,y,z data
                        for row in f:
                            vals = row.strip().split(",")
                            if len(vals) != 3:
                                continue
                            x, y, z = map(float, vals)
                            read_mesh(offset=(x, y, z))
                    self._pattern_data = ""
                else:
                    read_mesh()
                
            except Exception as e:
                ps.error(f"Error loading tetrahedral mesh:\n{e}")
            finally:
                root.destroy()

    def duplicate_tet_elastic_body(self, pattern_body: TetrahedralElastodynamicsBody, pattern_data: str):
        if self._pattern_data:
            with open(self._pattern_data, "r") as f:
                # this file is a csv with x, y, z coordinates. 
                # for each row, we want to load a mesh using read_mesh, offset using the x,y,z data
                for row in f:
                    vals = row.strip().split(",")
                    if len(vals) != 3:
                        continue
                    print(vals)
                    x, y, z = map(float, vals)
                    id = self._get_new_id()
                    body = TetrahedralElastodynamicsBody()
                    body.on_mesh_loaded(
                        f"{pattern_body.name} - {id}",
                        pattern_body.VT + np.array([x, y, z]),
                        pattern_body.T,
                        pattern_body.R,
                        pattern_body.Ye,
                        pattern_body.nue,
                        pattern_body.rhoe,
                        pattern_body.bext,
                        pattern_body.aext,
                        pattern_body.v0,
                        headless=False
                    )
                    self._transform_library.on_mesh_added(body.name)
                    self._tet_elastic_bodies.append(body)
                    
            self._pattern_data = ""
        

    def _load_pattern_file(self):
        root = tk.Tk()
        root.withdraw()
        self._pattern_data = filedialog.askopenfilename(
            title="Select pattern file",
            defaultextension=".csv",
            filetypes=[("Pattern files", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()

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
