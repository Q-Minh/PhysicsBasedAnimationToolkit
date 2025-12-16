# type: ignore
import polyscope as ps
import polyscope.imgui as imgui
import numpy as np
import scipy as sp

class PsHelper:
    _show_mesh: bool = True
    _show_gizmo: bool = False
    _mesh: ps.Structure

    def __init__(self, mesh:ps.Structure, show_mesh=True, show_gizmo=False):
        self._mesh = mesh
        self._show_gizmo = show_gizmo
        self._show_mesh = show_mesh

    def get_show_gizmo(self):
        return self._show_gizmo
    
    def get_show_mesh(self):
        return self._show_mesh

    def draw(self, as_tab_items=True):

        # Since visibility can be manipulated elsewhere, always get current value first
        self._show_mesh = self._mesh.is_enabled()
        self._show_gizmo = self._mesh.get_transform_gizmo_enabled()
        tab_flags = (
            imgui.ImGuiTabBarFlags_Reorderable
            | imgui.ImGuiTabBarFlags_FittingPolicyScroll
            | imgui.ImGuiTabBarFlags_TabListPopupButton
        )

        if as_tab_items:
            if imgui.BeginTabItem("Visibility", True, tab_flags)[0]:
                self._visibility_controls_draw()
                imgui.EndTabItem()

            if imgui.BeginTabItem("Fine Transform", True, tab_flags)[0]:
                self._fine_transfrom_controls_draw()
            
                if imgui.TreeNode("Actual transform matrix"):
                    self._transform_matrix_draw()
                    imgui.TreePop()
                imgui.EndTabItem()
        
        else:
            if imgui.TreeNode("Visibility", True, tab_flags)[0]:
                self._visibility_controls_draw()
                imgui.TreePop()

            if imgui.TreeNode("Fine Transforms", True, tab_flags)[0]:
                self._fine_transfrom_controls_draw()
            
                if imgui.TreeNode("Actual transform matrix"):
                    self._transform_matrix_draw()
                    imgui.TreePop()
                imgui.TreePop()


    def _visibility_controls_draw(self):
        imgui.Text("Show:")
        imgui.SameLine()

        _, self._show_mesh = imgui.Checkbox("Structure", self._show_mesh)
        self._mesh.set_enabled(self._show_mesh)

        if self._show_mesh:
            imgui.SameLine()
            _, self._show_gizmo = imgui.Checkbox("Gizmo", self._show_gizmo)

        self._mesh.set_transform_gizmo_enabled(
            self._show_mesh and self._show_gizmo
        )

    def _fine_transfrom_controls_draw(self):
        transform = self._mesh.get_transform()
        euler_angles = sp.spatial.transform.Rotation.from_matrix(transform[:3, :3]).as_euler(
            "xyz", degrees=True
        )
        r_updated, r = imgui.SliderFloat3("Rotation XYZ", euler_angles, -180.0, 180.0)
        t_updated, t = imgui.SliderFloat3("Translation", transform[:3, 3].T, -2, 2)
        if r_updated:
            transform[:3, :3] = sp.spatial.transform.Rotation.from_euler(
                "xyz", r, degrees=True
            ).as_matrix()

        if t_updated:
            transform[:3, 3] = np.array(t).T


        if t_updated or r_updated:    
            self._mesh.set_transform(transform)
    
    def _transform_matrix_draw(self):
        transform = self._mesh.get_transform()
        for row in range(4):                
            imgui.PushID(f"{self._mesh.get_name()}--{row}")
            _, transform[row, :] = imgui.InputFloat4("", transform[row, :])
            imgui.PopID()
        self._mesh.set_transform(transform)