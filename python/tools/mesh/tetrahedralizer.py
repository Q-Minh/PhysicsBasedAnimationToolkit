# type: ignore
import gpytoolbox as gpt
import igl
import numpy as np
import meshio
import argparse

import polyscope as ps
import polyscope.imgui as imgui
import numpy as np
import argparse
import meshio
import tetgen as tg
import tkinter as tk
from tkinter import filedialog
import fast_simplification as fs

from pbatoolkit import pbat, pypbat


def define_args():
    parser = argparse.ArgumentParser(description="Generate a beam mesh.")
    parser.add_argument(
        "--mode",
        type=str,
        default="obj",
        choices=["obj", "cyl", "prism"],
        help="Mode that will be opened. Choice of: "
        "obj - tetrahedralizes obj file given as input; "
        "cyl - create cylinder beam with defined equator and height divisions; "
        "prism - create prism beam with defined resolution",
    )
    parser.add_argument(
        "--dims",
        nargs="+",
        type=float,
        default=[1.0, 1.0, 1.0],
        help="Dimensions of the cylinder/prism in the format: dx dy dz",
    )
    parser.add_argument(
        "--equator_divisions",
        "--equator-divisions",
        type=int,
        default=40,
        help="For mode == cyl. Number of divisions around the equator.",
    )
    parser.add_argument(
        "--height_divisions",
        "--height-divisions",
        type=int,
        default=20,
        help="For mode == cyl. Number of divisions along the height of the cylinder.",
    )
    parser.add_argument(
        "--resolution",
        nargs="+",
        type=int,
        default=[40, 10, 10],
        help="For mode == prism. Mesh resolution for the beam in the format: nx ny nz",
    )
    parser.add_argument(
        "--min_ratio",
        "--min-ratio",
        type=float,
        default=1.1,
        help="Min ratio argument for TetGen.",
    )
    parser.add_argument(
        "--min_dihedral",
        "--min-dihedral",
        type=float,
        default=10.0,
        help="Min dihedral angle argument for TetGen.",
    )
    parser.add_argument(
        "--steiner_points",
        "--steiner-points",
        type=int,
        default=-1,
        help="Number of steiner points to use in TetGen. -1 for automatic.",
    )
    parser.add_argument(
        "--nobisect",
        action="store_true",
        help="Disable bisecting the mesh when tetrahedralizing.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="mesh-tetrahedralized.mesh",
        help="Output filename for the mesh.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input filename for the mesh. Required when mode == obj",
        dest="input",
    )
    parser.add_argument(
        "--normalize",
        type=bool,
        default=False,
        help="Normalize tetrahedralized mesh",
    )
    parser.add_argument(
        "-v",
        "--visual",
        type=bool,
        default=True,
        help="Initialize a polyscope environment for visualization.",
    )
    parser.add_argument(
        "--remesh",
        type=bool,
        default=False,
        help="Remesh the provided surface mesh before tetrahedralization.",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="Project remeshed vertices onto original mesh",
        type=bool,
        action=argparse.BooleanOptionalAction,
        dest="project",
        default=True,
    )
    parser.add_argument(
        "-k",
        "--iterations",
        help="Number of remeshing iterations",
        type=int,
        dest="iterations",
        default=10,
    )
    parser.add_argument(
        "-f",
        "--feature-dihedral-angle",
        help="Minimum dihedral angle for an edge to be considered sharp",
        type=float,
        dest="feature_dihedral",
        default=0.1,
    )
    parser.add_argument(
        "--remesh-edge-length",
        help="Target edge length for remeshing",
        type=float,
        dest="remesh_edge_length",
        default=0.1,
    )
    parser.add_argument(
        "-t",
        "--target-reduction",
        help="Target reduction in mesh size in percentage",
        type=float,
        dest="target_reduction",
        default=0.9,
    )
    parser.add_argument(
        "-a",
        "--aggressiveness",
        help="High value means sacrifice quality for speed, low value means sacrifice speed for quality",
        type=int,
        dest="aggressiveness",
        default=5,
    )
    return parser.parse_args()


def make_cylinder(equator_divisions, height_divisions, dims, normalize):

    V, F = gpt.cylinder(equator_divisions, height_divisions)
    # Make the mesh fit in a unit cube (one circular face is a x = 0 and the other at x = -1)
    V[:, :2] /= 2

    # Center the mesh at the origin
    V -= np.array([0, 0, 0.5])

    # Scale the mesh by the desired dimensions
    V *= np.array(dims)

    if normalize:
        max_dim = max(dims)
        V /= max_dim

    R_y = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
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
            F = np.vstack((F, np.array([start + j, start + next_j, center_index])))
    return V, F


def make_prism(resolution, dims, normalize):
    nx, ny, nz = resolution

    placement = np.linspace(-0.5, 0.5, nz)
    V = None
    F = None
    new_tris = None
    for i in range(nz):
        v, f = gpt.regular_square_mesh(nx, ny)
        # make it unit length
        v /= 2
        # add 3rd dimension to V
        v = np.hstack((v, np.full((v.shape[0], 1), placement[i])))
        if V is None:
            V = v
            F = f
            continue

        f += V.shape[0]
        F = np.vstack((F, f))
        V = np.vstack((V, v))

        # Add tets between layers (will be added to main list after loop, to avoid thowing off indices)
        cur_row = ny * nx * i
        last_row = ny * nx * (i - 1)
        cur_row_end = ny * nx * (i + 1) - nx
        last_row_end = ny * nx * (i) - nx
        for j in range(nx - 1):
            for cur, last in zip([cur_row, cur_row_end], [last_row, last_row_end]):
                cur_j = cur + j
                next_j = cur_j + 1
                below_j = last + j
                below_next_j = below_j + 1

                next_tris = np.array(
                    [[cur_j, below_j, below_next_j], [cur_j, below_next_j, next_j]]
                )
                if new_tris is None:
                    new_tris = next_tris
                    continue
                new_tris = np.vstack((new_tris, next_tris))

        cur_col = ny * nx * i
        last_col = ny * nx * (i - 1)
        cur_col_end = ny * nx * i + (nx - 1)
        last_col_end = ny * nx * (i - 1) + (nx - 1)

        for j in range(ny - 1):
            for cur, last in zip([cur_col, cur_col_end], [last_col, last_col_end]):
                cur_j = cur + j * nx
                next_j = cur + (j + 1) * nx
                below_j = last + j * nx
                below_next_j = last + (j + 1) * nx

                next_tris = np.array(
                    [[cur_j, below_j, below_next_j], [cur_j, below_next_j, next_j]]
                )
                if new_tris is None:
                    new_tris = next_tris
                    continue
                new_tris = np.vstack((new_tris, next_tris))

    F = np.vstack((F, new_tris))

    # Scale the mesh by the desired dimensions
    V *= np.array(dims)
    if normalize:
        max_dim = max(dims)
        V /= max_dim

    R_y = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    V = V @ R_y

    return V, F


def remesh(V, T, args):
    SE, E, uE, Emap, uE2E, sharp = igl.sharp_edges(V, T, args.feature_dihedral)
    uEsharp = uE[sharp, :]
    feature_vertices = np.unique(uEsharp)
    Vr, Tr = gpt.remesh_botsch(
        V,
        T,
        i=args.iterations,
        project=args.project,
        h=args.remesh_edge_length,
        feature=feature_vertices,
    )
    return Vr, Tr


def simplify(V, T, args):
    V, T = fs.simplify(
        V, T, target_reduction=args.target_reduction, agg=args.aggressiveness
    )
    return V, T


def to_tets(V, T, args):
    tgen = tg.TetGen(V, T)
    nodes, elem, attrib = tgen.tetrahedralize(
        steinerleft=args.steiner_points,
        minratio=args.min_ratio,
        mindihedral=args.min_dihedral,
        nobisect=args.nobisect,
        regionattrib=True,
    )
    return nodes, elem, attrib


def set_volume_mesh(nodes, elem, attrib):
    # Show determinant
    jac = pbat.fem.determinant_of_jacobian(
        elem.T, nodes.T, pbat.fem.Element.Tetrahedron, order=1, quadrature_order=1
    )

    vm = ps.register_volume_mesh("Volume", nodes, elem, edge_width=1.0)
    vm.add_scalar_quantity("Regions", attrib.ravel(), defined_on="cells", enabled=True)
    vm.add_scalar_quantity("Jacobian", jac.ravel(), defined_on="cells", enabled=False)
    return vm


if __name__ == "__main__":
    args = define_args()

    nodes, elem, attrib = None, None, None

    if args.mode == "cyl":
        V, T = make_cylinder(
            equator_divisions=args.equator_divisions,
            height_divisions=args.height_divisions,
            dims=args.dims,
            normalize=args.normalize,
        )

    elif args.mode == "prism":
        V, T = make_prism(
            resolution=args.resolution, dims=args.dims, normalize=args.normalize
        )

    else:
        # Load mesh from file
        imesh = meshio.read(args.input)
        V = imesh.points
        # Assume first cell block is the surface
        if "triangle" not in imesh.cells_dict:
            raise ValueError("Mesh must contain triangle faces.")
        T = imesh.cells_dict["triangle"]

    if not args.visual:
        if args.remesh:
            V, T = remesh(V, T, args)
        nodes, elem, attrib = to_tets(V, T, args)
        omesh = meshio.Mesh(
            nodes, [("tetra", elem)], cell_data={"medit:ref": [attrib.ravel()]}
        )
        meshio.write(args.output, omesh)
        exit(0)

    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Tetrahedralizer")
    ps.init()

    # remeshed version of loaded mesh
    Vr, Tr = V, T

    sm = ps.register_surface_mesh("Surface", V, T, edge_width=1.0)
    sm2 = ps.register_surface_mesh("Remeshed Surface", Vr, Tr, edge_width=1.0)
    sm2.set_position(np.array([1.5, 0.0, 0.0]))

    vm = None

    def callback():
        global vm, sm, sm2, args
        global V, T, Vr, Tr, nodes, elem, attrib

        if args.mode == "cyl":
            a, args.equator_divisions = imgui.InputInt(
                "Equator divisions", args.equator_divisions
            )
            b, args.height_divisions = imgui.InputInt(
                "Height divisions", args.height_divisions
            )
            c, args.dims = imgui.InputFloat3("Dimensions", args.dims)
            d, args.normalize = imgui.Checkbox("Normalize", args.normalize)
            if (
                (a or b or c or d)
                and args.height_divisions > 2
                and args.equator_divisions > 3
            ):
                transform = sm.get_transform()
                V, T = make_cylinder(
                    equator_divisions=args.equator_divisions,
                    height_divisions=args.height_divisions,
                    dims=args.dims,
                    normalize=args.normalize,
                )
                sm = ps.register_surface_mesh("Surface", V, T, edge_width=1.0)
                sm.set_transform(transform)

        elif args.mode == "prism":
            a, args.resolution = imgui.InputInt3("Resolution", args.resolution)
            c, args.dims = imgui.InputFloat3("Dimensions", args.dims)
            d, args.normalize = imgui.Checkbox("Normalize", args.normalize)
            if (
                (a or c or d)
                and args.resolution[0] > 2
                and args.resolution[1] > 2
                and args.resolution[2] > 2
            ):
                transform = sm.get_transform()
                V, T = make_prism(
                    resolution=args.resolution, dims=args.dims, normalize=args.normalize
                )
                sm = ps.register_surface_mesh("Surface", V, T, edge_width=1.0)
                sm.set_transform(transform)

        if imgui.TreeNode("TetGen Parameters"):
            _, args.min_ratio = imgui.InputFloat("Minimal ratio", args.min_ratio)
            _, args.min_dihedral = imgui.InputFloat(
                "Minimal dihedral angle", args.min_dihedral
            )
            _, args.steiner_points = imgui.InputInt(
                "Steiner points (-1 for auto)", args.steiner_points
            )
            _, args.nobisect = imgui.Checkbox("Disable bisecting mesh", args.nobisect)
            imgui.TreePop()

        if imgui.TreeNode("Remeshing Parameters"):
            a, args.remesh_edge_length = imgui.InputFloat(
                "Target edge length", args.remesh_edge_length
            )
            b, args.iterations = imgui.InputInt("Remeshing iterations", args.iterations)
            c, args.feature_dihedral = imgui.InputFloat(
                "Feature dihedral angle", args.feature_dihedral
            )
            d, args.project = imgui.Checkbox("Project onto original mesh", args.project)

            if imgui.Button("Remesh Surface"):
                Vr, Tr = remesh(V, T, args)
                transform = sm2.get_transform()
                sm2 = ps.register_surface_mesh(
                    "Remeshed Surface", Vr, Tr, edge_width=1.0
                )
                sm2.set_transform(transform)
            imgui.TreePop()

        if imgui.Button("Tetrahedralize"):
            if vm is not None:
                transform = vm.get_transform()
            else:
                transform = sm.get_transform()
                transform[:3, 3] = np.array([0.0, 0.0, 1.5])
            nodes, elem, attrib = to_tets(Vr, Tr, args)
            vm = set_volume_mesh(nodes, elem, attrib)
            vm.set_transform(transform)

        if imgui.Button("Save"):
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.asksaveasfilename(
                title="Save session (MESH)",
                defaultextension=".mesh",
                filetypes=[
                    ("Tetrahedral mesh files (ASCII)", "*.mesh"),
                    ("Tetrahedral mesh files (binary)", "*.msh"),
                    ("All files", "*.*"),
                ],
            )
            try:
                if file_path:
                    omesh = meshio.Mesh(
                        nodes,
                        [("tetra", elem)],
                        cell_data={"medit:ref": [attrib.ravel()]},
                    )
                    meshio.write(file_path, omesh)
            except Exception as e:
                ps.error(f"Error saving session:\n{e}")
            finally:
                root.destroy()

    ps.set_user_callback(callback)
    ps.show()
