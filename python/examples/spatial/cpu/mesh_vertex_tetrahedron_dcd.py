import pbatoolkit as pbat
import argparse
import meshio
import numpy as np
import polyscope as ps
import polyscope.imgui as imgui
import igl
import itertools


def combine(V: list, C: list):
    Vsizes = [Vi.shape[0] for Vi in V]
    offsets = list(itertools.accumulate(Vsizes))
    C = [C[i] + offsets[i] - Vsizes[i] for i in range(len(C))]
    C = np.vstack(C)
    V = np.vstack(V)
    return V, C


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="3D multibody tetrahedral mesh DCD contact detection algorithm",
    )
    parser.add_argument(
        "-i", "--input", help="Path to input mesh", dest="input", required=True
    )
    parser.add_argument(
        "--rows", help="Number of rows in the grid", dest="rows", type=int, default=4
    )
    parser.add_argument(
        "--cols", help="Number of columns in the grid", dest="cols", type=int, default=4
    )
    parser.add_argument(
        "--height", help="Height of the grid", dest="height", type=int, default=2
    )
    parser.add_argument(
        "--separation",
        help="Separation between meshes as percentage of input mesh bounding box extents",
        dest="separation",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--frequency",
        help="Frequency of the oscillation",
        dest="frequency",
        type=float,
        default=1.2,
    )
    parser.add_argument(
        "--amplitude",
        help="Amplitude of the oscillation",
        dest="amplitude",
        type=float,
        default=1.2,
    )
    args = parser.parse_args()

    # Load input mesh
    imesh = meshio.read(args.input)
    V, C = imesh.points.astype(np.float64), imesh.cells_dict["tetra"].astype(np.int64)

    # Duplicate mesh into grid of meshes
    grows = args.rows
    gcols = args.cols
    gheight = args.height
    aabb = pbat.geometry.aabb(V.T)
    extents = aabb.max - aabb.min
    separation = args.separation * extents
    VS = [
        V
        + np.array(
            [
                i * (extents[0] + separation[0]),
                j * (extents[1] + separation[1]),
                k * (extents[2] + separation[2]),
            ]
        )
        for k in range(gheight)
        for j in range(gcols)
        for i in range(grows)
    ]
    CS = [C for _ in range(gheight) for _ in range(gcols) for _ in range(grows)]
    V, C = combine(VS, CS)
    X, T = V.T, C.T
    mcd = pbat.sim.contact.MeshVertexTetrahedronDcd(X, T)
    VP = mcd.multibody_system.VP
    V = mcd.multibody_system.V
    F = mcd.multibody_system.F
    U = np.zeros_like(X)
    XT = X.copy()
    XTP1 = X.copy()

    ps.init()
    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("3D multibody mesh CCD and DCD contact detection algorithm")

    sm = ps.register_surface_mesh("Surface mesh", X.T, F.T)
    sm.set_transparency(0.5)
    sm.set_edge_width(1)

    t = 0
    animate = False
    export = False
    n_body_pairs = 0
    n_contact_pairs = 0
    contacting_triangles = np.zeros(F.shape[1], dtype=np.float32)
    profiler = pbat.profiling.Profiler()

    def callback():
        global t, animate, export
        global U, XT, XTP1, n_body_pairs, n_contact_pairs, contacting_triangles

        changed, animate = imgui.Checkbox("Animate", animate)
        step = imgui.Button("Step")

        changed, export = imgui.Checkbox("Export", export)

        if animate or step:
            if export:
                ps.screenshot()

            profiler.begin_frame("Physics")
            XT = XTP1
            U[:] = 0
            for m in range(VP.shape[0] - 1):
                r = m % 3
                # Update mesh position
                offset = args.separation * extents
                dir = np.ones(3)
                dir[r] = -1
                u = args.amplitude * dir * offset * np.sin(t * args.frequency)
                U[:, V[VP[m] : VP[m + 1]]] = u[:, np.newaxis]
            XTP1 = X + U
            mcd.update_active_set(XTP1, T)
            vfc = mcd.vertex_triangle_contacts()
            v, f = vfc[0, :], vfc[1, :]
            profiler.end_frame("Physics")

            contacting_triangles[:] = 0
            contacting_triangles[f] = 1
            n_contact_pairs = vfc.shape[1]
            ps.register_point_cloud("DCD verts", XTP1[:, V[v]].T)
            sm.update_vertex_positions(XTP1.T)
            sm.add_scalar_quantity(
                "Contacting triangles",
                contacting_triangles,
                defined_on="faces",
                enabled=True,
                cmap="coolwarm",
                vminmax=(0, 1),
            )

            t = t + 1 / 60

        imgui.Text(f"# of contact pairs={n_contact_pairs}")

    ps.set_user_callback(callback)
    ps.show()
