# type: ignore
import meshio
import polyscope as ps
import polyscope.imgui as imgui
import tkinter as tk
from tkinter import filedialog
import numpy as np

if __name__ == "__main__":
    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Transform")
    ps.init()
    vm: ps.VolumeMesh = None
    sm: ps.SurfaceMesh = None
    V, F, C = None, None, None

    def callback():
        global vm, sm
        global V, F, C
        if imgui.Button("Load", [imgui.GetWindowWidth() / 2.1, 0]):
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(
                title="Select mesh file",
                defaultextension=".mesh",
                filetypes=[
                    ("Tetrahedral mesh files (ASCII)", "*.mesh"),
                    ("Tetrahedral mesh files (binary)", "*.msh"),
                    ("Triangle mesh wavefront", "*.obj"),
                    ("Triangle mesh PLY", "*.ply"),
                    ("Triangle mesh (binary)", "*.stl"),
                    ("All files", "*.*"),
                ],
            )
            if file_path:
                imesh = meshio.read(file_path)
                if "tetra" in imesh.cells_dict:
                    V, C = imesh.points, imesh.cells_dict["tetra"]
                    vm = ps.register_volume_mesh("Mesh", V, C)
                    vm.set_transform(np.eye(4))
                elif "triangle" in imesh.cells_dict:
                    V, F = imesh.points, imesh.cells_dict["triangle"]
                    sm = ps.register_surface_mesh("Mesh", V, F)
                    sm.set_transform(np.eye(4))
            root.destroy()

        if imgui.Button("Save", [imgui.GetWindowWidth() / 2.1, 0]):
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.asksaveasfilename(
                title="Select mesh file",
                defaultextension=".mesh",
                filetypes=[
                    ("Tetrahedral mesh files (ASCII)", "*.mesh"),
                    ("Tetrahedral mesh files (binary)", "*.msh"),
                    ("Triangle mesh wavefront", "*.obj"),
                    ("Triangle mesh PLY", "*.ply"),
                    ("Triangle mesh (binary)", "*.stl"),
                    ("All files", "*.*"),
                ],
            )
            if file_path:
                if vm is not None:
                    T = vm.get_transform()
                    VH = np.vstack([V.T, np.ones((1, V.shape[0]))])
                    VT = (T @ VH).T[:, :3]
                    omesh = meshio.Mesh(VT, [("tetra", C)])
                    meshio.write(file_path, omesh)
                if sm is not None:
                    T = sm.get_transform()
                    VH = np.vstack([V.T, np.ones((1, V.shape[0]))])
                    VT = (T @ VH).T[:, :3]
                    omesh = meshio.Mesh(VT, [("triangle", F)])
                    meshio.write(file_path, omesh)
            root.destroy()

        if V is not None:
            T = (
                vm.get_transform()
                if vm is not None
                else sm.get_transform() if sm is not None else np.eye(4)
            )
            VH = np.vstack([V.T, np.ones((1, V.shape[0]))])
            VT = (T @ VH).T[:, :3]
            xmin = VT.min(axis=0)
            xmax = VT.max(axis=0)
            extents = xmax - xmin
            imgui.Text(f"Bounding box min: {xmin}")
            imgui.Text(f"Bounding box max: {xmax}")
            imgui.Text(f"Bounding box extents: {extents}")


    ps.set_user_callback(callback)
    ps.show()
