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
        "--resolution",
        nargs="+",
        type=int,
        default=[40, 10, 10],
        help="Mesh resolution for the beam in the format: nx ny nz",
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
        default="prism-new.mesh",
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


def make_prism(resolution,
                  dims, 
                  normalize):
    nx, ny, nz = resolution
    
    placement = np.linspace(0, 1, nz)
    V = None
    T = None
    new_tets = None
    for i in range(nz):
        v, t = gpt.regular_square_mesh(nx, ny)
        # make it unit length
        v /= 2
        # add 3rd dimension to V
        v = np.hstack((v, np.full((v.shape[0], 1), placement[i])))
        if V is None:
            V = v
            T = t
            continue
        
        t += V.shape[0]    
        T = np.vstack((T, t))
        V = np.vstack((V, v))

        # Add tets between layers (will be added to main list after loop, to avoid thowing off indices)
        cur_row = ny * nx * i
        last_row = ny * nx * (i - 1)
        cur_row_end = ny * nx * (i+1) - nx
        last_row_end = ny * nx * (i) - nx
        for j in range(nx-1):
            for cur, last in zip([cur_row, cur_row_end],[last_row, last_row_end]):
                cur_j = cur + j
                next_j = cur_j + 1
                below_j = last + j
                below_next_j = below_j + 1
                

                next_tris = np.array([
                        [cur_j, below_j, below_next_j],
                        [cur_j, below_next_j, next_j]
                    ])
                if new_tets is None:
                    new_tets = next_tris
                    continue
                new_tets = np.vstack((new_tets, next_tris))
        
        cur_col = ny * nx * i
        last_col = ny * nx * (i - 1)
        cur_col_end = ny * nx * i + (nx - 1)
        last_col_end = ny * nx * (i - 1) + (nx - 1)

        for j in range(ny-1):
            for cur, last in zip([cur_col, cur_col_end],[last_col, last_col_end]):
                cur_j = cur + j * nx
                next_j = cur + (j + 1) * nx
                below_j = last + j * nx
                below_next_j = last + (j+1) * nx
                

                next_tris = np.array([
                        [cur_j, below_j, below_next_j],
                        [cur_j, below_next_j, next_j]
                    ])
                if new_tets is None:
                    new_tets = next_tris
                    continue
                new_tets = np.vstack((new_tets, next_tris))

    T = np.vstack((T, new_tets))
        

    # Scale the mesh by the desired dimensions
    V *= np.array(dims)
    if normalize:
        max_dim = max(dims)
        V /= max_dim

    return V, T


if __name__ == "__main__":
    args = define_args()

    
    V, T = make_prism(
        resolution=args.resolution,
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
    

    dims = args.dims
    normalize = args.normalize
    min_ratio = args.min_ratio
    resolution = args.resolution
    def callback():
        global vm, sm, dims, normalize, min_ratio, resolution
        global V, T, nodes, elem

        a, resolution = imgui.InputInt3("Equator divisions", resolution)
        c, dims = imgui.InputFloat3("Dimensions", dims)
        _, min_ratio = imgui.InputFloat("Minimal ratio", min_ratio)
        d, normalize = imgui.Checkbox("Normalize", normalize)
        if (a or c or d) and resolution[0]>2 and resolution[1]>2 and resolution[2]>2:
            transform = sm.get_transform()
            V, T = make_prism(
                resolution=resolution,
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


    






