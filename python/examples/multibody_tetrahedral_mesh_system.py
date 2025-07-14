import pbatoolkit as pbat
import meshio
import numpy as np
import polyscope as ps
import polyscope.imgui as imgui
import itertools
import argparse


def combine(V: list, C: list):
    Vsizes = [Vi.shape[0] for Vi in V]
    offsets = list(itertools.accumulate(Vsizes))
    C = [C[i] + offsets[i] - Vsizes[i] for i in range(len(C))]
    C = np.vstack(C)
    V = np.vstack(V)
    return V, C


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Multibody Tetrahedral Mesh System",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Path to input mesh",
        nargs="+",
        dest="inputs",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--translation",
        help="Translation factor for the meshes",
        type=float,
        default=0.1,
        dest="translation",
        required=False,
    )
    args = parser.parse_args()

    imeshes = [meshio.read(input) for input in args.inputs]
    V, C = [
        imesh.points / (imesh.points.max() - imesh.points.min()) for imesh in imeshes
    ], [imesh.cells_dict["tetra"] for imesh in imeshes]
    for i in range(len(V) - 1):
        extent = V[i][:, -1].max() - V[i][:, -1].min()
        offset = V[i][:, -1].max() - V[i + 1][:, -1].min()
        V[i + 1][:, -1] += offset + extent * args.translation
    V, C = combine(V, C)
    V = V.astype(np.float64)
    C = C.astype(np.int64)
    X, T = V.T, C.T

    # Mess up the ordering of the tets to test robustness of
    # pbat.sim.contact.MultibodyTetrahedralMeshSystem
    Tswap = np.array(T[:, 0])
    T[:, 0] = T[:, -1]
    T[:, -1] = Tswap

    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Multibody Tetrahedral Mesh System")
    ps.init()

    meshes = pbat.sim.contact.MultibodyTetrahedralMeshSystem()
    meshes.Construct(X, T)
    n_bodies = meshes.n_bodies
    for o in range(n_bodies):
        Vo = meshes.vertices_of(o)
        ps.register_point_cloud(
            f"Body {o} vertices",
            X[:, Vo].T,
            radius=0.01,
            transparency=0.5,
        )
        Fo = meshes.triangles_of(o)
        ps.register_surface_mesh(
            f"Body {o} triangles", X.T, Fo.T, transparency=0.5, edge_width=1
        )
        To = meshes.tetrahedra_of(o, T)
        ps.register_volume_mesh(
            f"Body {o} tetrahedral mesh",
            X.T,
            To.T,
            transparency=0.5,
            edge_width=1,
            enabled=False,
        )

    ps.register_point_cloud("Contact vertices", X[:, meshes.V].T, enabled=False)
    ps.register_surface_mesh("Contact triangles", X.T, meshes.F.T, enabled=False)
    ps.register_volume_mesh("Tetrahedral Mesh", X.T, T.T, enabled=False)
    ps.show()
