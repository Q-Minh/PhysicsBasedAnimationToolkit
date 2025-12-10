import polyscope as ps
import polyscope.imgui as imgui
import numpy as np
import scipy as sp

def rot_x(phi):
    c = np.cos(phi); s = np.sin(phi)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rot_y(theta):
    c = np.cos(theta); s = np.sin(theta)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rot_z(psi):
    c = np.cos(psi); s = np.sin(psi)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def to_rad(deg):
    return deg * np.pi / 180

def to_deg(rad):
    return rad * 180 / np.pi

def decompose_zyx(R, eps=1e-8):
    """
    Decompose R into yaw(psi)-pitch(theta)-roll(phi) such that
    R = Rz(psi) @ Ry(theta) @ Rx(phi).
    Returns (phi, theta, psi) in radians and the three rotation matrices (Rx, Ry, Rz).
    """
    # numeric safety
    assert R.shape == (3,3), "R must be 3x3"
    # clip input to asin to [-1,1] to avoid nan from tiny numerical over/underflow
    r31 = R[2,0]
    theta = np.arcsin(np.clip(-r31, -1.0, 1.0))

    # cos(theta) might be tiny -> check for gimbal lock
    ctheta = np.cos(theta)
    if abs(ctheta) > eps:
        phi = np.arctan2(R[2,1], R[2,2])     # roll
        psi = np.arctan2(R[1,0], R[0,0])     # yaw
    else:
        # Gimbal lock: cos(theta) ~= 0
        # choose phi = 0 and solve for psi
        phi = 0.0
        psi = np.arctan2(-R[0,1], R[1,1])

    
    return phi, theta, psi


class PsHelper:
    _show_mesh: bool = True
    _show_gizmo: bool = False
    _mesh: ps.Structure

    def __init__(self, mesh:ps.Structure, show_mesh=True, show_gizmo=False):
        self._mesh = mesh
        self._show_gizmo = show_gizmo
        self._show_mesh = show_mesh

    def draw(self):

        # Since visibility can be manipulated elsewhere, always get current value first
        self._show_mesh = self._mesh.is_enabled()
        self._show_gizmo = self._mesh.get_transform_gizmo_enabled()

        _, self._show_mesh = imgui.Checkbox("Visible", self._show_mesh)
        self._mesh.set_enabled(self._show_mesh)

        if self._show_mesh:
            imgui.SameLine()
            _, self._show_gizmo = imgui.Checkbox("Gizmo", self._show_gizmo)

        self._mesh.set_transform_gizmo_enabled(
            self._show_mesh and self._show_gizmo
        )

        if imgui.TreeNode("Fine transform controls. Beware of gimbal lock!"):
            self.on_fine_transfrom_controls()
            
            if imgui.TreeNode("Actual transform matrix"):
                transform = self._mesh.get_transform()
                for row in range(4):                
                    imgui.PushID(f"{self._mesh.get_name()}--{row}")
                    _, transform[row, :] = imgui.InputFloat4("", transform[row, :])
                    imgui.PopID()
                self._mesh.set_transform(transform)
                imgui.TreePop()
                
            imgui.TreePop()

    def on_fine_transfrom_controls(self):
        transform = self._mesh.get_transform()
        euler_angles = sp.spatial.transform.Rotation.from_matrix(transform[:3, :3]).as_euler(
            "xyz", degrees=True
        )
        r_updated, r = imgui.SliderFloat3("Rotation XYZ", euler_angles, -180.0, 180.0)
        if r_updated:
            transform[:3, :3] = sp.spatial.transform.Rotation.from_euler(
                "xyz", r, degrees=True
            ).as_matrix()
            self._mesh.set_transform(transform)
    