import gpytoolbox as gpt
import numpy as np
import meshio
import argparse

from pbatoolkit import pbat, pypbat
import igl
import polyscope as ps
import polyscope.imgui as imgui
import polyscope.implot as implot
import numpy as np
import scipy as sp
import argparse
import meshio
import scipy
import tetgen as tg
import tkinter as tk
from tkinter import filedialog

def define_args():
    parser = argparse.ArgumentParser(description="Generate a beam mesh.")
    parser.add_argument(
        "--dims",
        nargs="+",
        type=float,
        default=[1.0, 1.0, 1.0],
        help="Dimensions of the beam in the format: dx dy dz",
    )
    parser.add_argument(
        "--equator_divisions",
        type=int,
        default=40,
        help="Number of divisions around the equator.",
    )
    parser.add_argument(
        "--height_divisions",
        type=int,
        default=20,
        help="Number of divisions along the height of the cylinder.",
    )
    parser.add_argument(
        "--min_ratio",
        type=float,
        default=1.1,
        help="Min ratio argument for TetGen.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="spaghetti-new.mesh",
        help="Output filename for the mesh.",
    )
    parser.add_argument(
        "--normalize",
        type=bool,
        default=False,
        help="Normalize cylinder",
    )
    parser.add_argument(
        "-v",
        "--visual",
        type=bool,
        default=True,
        help="Initialize a polyscope environment for visualization.",
    )
    return parser.parse_args()


def make_cylinder(equator_divisions, 
                  height_divisions,
                  dims, 
                  normalize):
    
    V, T = gpt.cylinder(equator_divisions, height_divisions)
    # Make the mesh fit in a unit cube
    V[:, :2] /= 2

    # Scale the mesh by the desired dimensions
    V *= np.array(dims)
    if normalize:
        max_dim = max(dims)
        V /= max_dim

    R_y = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ])
    V = V @ R_y


    # Create planar caps at each planar division
    for i in range(height_divisions):
        start = i * equator_divisions
        stop = (i + 1) * equator_divisions

        # Triangulate interior faces
        # TODO: it might be better to seed more points to make better triangles. This will help tetgen make more regular tets on the caps
        center = np.mean(V[start:stop, :], axis=0)
        V = np.vstack((V, center))

        center_index = V.shape[0] - 1
        for j in range(equator_divisions):
            next_j = (j + 1) % equator_divisions
            T = np.vstack((T, np.array([start + j, start + next_j, center_index])))
    return V, T


if __name__ == "__main__":
    args = define_args()
    
    V, T = make_cylinder(
        equator_divisions=args.equator_divisions,
        height_divisions=args.height_divisions,
        dims=args.dims,
        normalize=args.normalize
    )
    tgen = tg.TetGen(V, T)
    nodes, elem = tgen.tetrahedralize(steinerleft=-1, minratio=args.min_ratio)
    
    if not args.visual:
        

        omesh = meshio.Mesh(nodes, [("tetra", elem)])
        meshio.write(args.output, omesh)
        exit(0)


    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Spaghett maker")
    ps.init()

    sm = ps.register_surface_mesh("Surface", V, T, edge_width=1.0)
    sm.set_position(np.array([0.0, 0.0, 1.5]))
    vm = ps.register_volume_mesh("Volume", nodes, elem, edge_width=1.0)


    equator_divisions = args.equator_divisions
    height_divisions = args.height_divisions
    dims = args.dims
    normalize = args.normalize
    min_ratio = args.min_ratio
    def callback():
        global vm, sm, equator_divisions, height_divisions, dims, normalize, min_ratio
        global V, T, nodes, elem

        a, equator_divisions = imgui.InputInt("Equator divisions", equator_divisions)
        b, height_divisions = imgui.InputInt("Height divisions", height_divisions)
        c, dims = imgui.InputFloat3("Dimensions", dims)
        _, min_ratio = imgui.InputFloat("Minimal ratio", min_ratio)
        d, normalize = imgui.Checkbox("Normalize", normalize)
        if (a or b or c or d) and (height_divisions > 2 and equator_divisions > 3):
            transform = sm.get_transform()
            V, T = make_cylinder(
                equator_divisions=equator_divisions,
                height_divisions=height_divisions,
                dims=dims,
                normalize=normalize
            )
            sm = ps.register_surface_mesh("Surface", V, T, edge_width=1.0)
            sm.set_transform(transform)

        if imgui.Button("Tetrahedralize"):
            tgen = tg.TetGen(V, T)
            nodes, elem = tgen.tetrahedralize(steinerleft=-1, minratio=min_ratio)
            vm = ps.register_volume_mesh("Volume", nodes, elem, edge_width=1.0)
            

        if imgui.Button("Save"):
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.asksaveasfilename(
                title="Save session (HDF5)",
                defaultextension=".mesh",
                filetypes=[
                    ("Tetrahedral mesh files (ASCII)", "*.mesh"),
                    ("Tetrahedral mesh files (binary)", "*.msh"),
                    ("All files", "*.*"),
                ],
            )
            try:
                if file_path:
                    omesh = meshio.Mesh(nodes, [("tetra", elem)])
                    meshio.write(file_path, omesh)
            except Exception as e:
                ps.error(f"Error saving session:\n{e}")
            finally:
                root.destroy()

            

    ps.set_user_callback(callback)
    ps.show()


    






