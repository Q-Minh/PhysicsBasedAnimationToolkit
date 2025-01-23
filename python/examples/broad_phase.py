import pbatoolkit as pbat
import argparse
import meshio
import numpy as np
import polyscope as ps
import polyscope.imgui as imgui

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="3D broad phase algorithms",
    )
    parser.add_argument(
        "-i", "--input", help="Path to input mesh", dest="input", required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output",
        dest="output",
        required=False,
        default=".",
    )
    parser.add_argument(
        "-t",
        "--translation",
        help="Vertical translation",
        type=float,
        dest="translation",
        default=0.1,
    )
    args = parser.parse_args()

    imesh = meshio.read(args.input)
    V, C = imesh.points.astype(np.float64), imesh.cells_dict["tetra"].astype(np.int64)

    # Duplicate input mesh into 2 separate meshes
    n_tets = C.shape[0]
    n_pts = V.shape[0]
    parent = np.hstack(
        [np.zeros(n_tets, dtype=np.int32), np.ones(n_tets, dtype=np.int32)]
    )
    parent = pbat.gpu.common.Buffer(parent)
    C = np.vstack([C, C + V.shape[0]]).astype(np.int32)
    sap = pbat.gpu.geometry.SweepAndPrune(C.shape[0], 24 * C.shape[0])
    bvh = pbat.gpu.geometry.Bvh(C.shape[0], 24 * C.shape[0])
    dims = V.shape[1]
    aabbs = pbat.gpu.geometry.Aabb(dims, C.shape[0])
    profiler = pbat.profiling.Profiler()

    # Setup animation
    V = np.vstack([V, V])
    height = 3
    vimin = V[:n_pts, -1].argmin()
    vimax = V[:n_pts, -1].argmax()
    zmin = V[vimin, -1]
    zextent = V[vimax, -1] - zmin
    zmax = (height - 1) * zextent + zmin
    V[n_pts:, -1] = V[n_pts:, -1] + (height - 2) * zextent
    direction = [1, -1]
    min, max = np.min(V, axis=0), np.max(V, axis=0)
    min[-1] = zmin
    max[-1] = zmax

    # Setup GUI
    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Broad Phase Collision Detection")
    ps.init()

    t = 0
    speed = 0.01
    animate = False
    dhat = 0.0
    algorithms = [
        "Sweep and Prune",
        "Bounding Volume Hierarchy",
    ]
    # NOTE:
    # VE contains vertex positions of each element's vertices.
    # Each column of VE represents an element vertex.
    # Each block of 3 rows represents the x, y, z coordinates of each vertex (per column).
    # This format makes it easy to compute min/max bounds on each element row-wise.
    VE = np.zeros((dims * C.shape[0], C.shape[1]))
    L, U = np.zeros((dims, C.shape[0])), np.zeros((dims, C.shape[0]))
    overlapping = np.zeros(aabbs.n_boxes)
    algorithm = algorithms[0]
    export = False
    vm = ps.register_volume_mesh(f"Meshes", V, C)

    def callback():
        global dhat, VE, L, U
        global t, speed, animate, overlapping, algorithms, algorithm, export

        changed = imgui.BeginCombo("Algorithm", algorithm)
        if changed:
            for i in range(len(algorithms)):
                _, selected = imgui.Selectable(
                    algorithms[i], algorithm == algorithms[i]
                )
                if selected:
                    algorithm = algorithms[i]
            imgui.EndCombo()
        changed, dhat = imgui.InputFloat("Box expansion", dhat, format="%.4f")
        changed, speed = imgui.InputFloat("Speed", speed, format="%.4f")
        changed, export = imgui.Checkbox("Export", export)
        changed, animate = imgui.Checkbox("Animate", animate)
        step = imgui.Button("Step")

        if animate or step:
            if export:
                ps.screenshot(f"{args.output}/{t}.png")

            profiler.begin_frame("Physics")
            for i in range(2):
                if V[i * n_pts + vimax, -1] >= zmax:
                    direction[i] = -1
                if V[i * n_pts + vimin, -1] <= zmin:
                    direction[i] = 1
                V[i * n_pts : (i + 1) * n_pts, -1] = (
                    V[i * n_pts : (i + 1) * n_pts, -1] + direction[i] * speed
                )
            vm.update_vertex_positions(V)
            for d in range(C.shape[1]):
                VE[:, d] = V[C[:, d], :].flatten(order="C")
            L = np.min(VE, axis=1).reshape((dims, C.shape[0]), order="F")
            U = np.max(VE, axis=1).reshape((dims, C.shape[0]), order="F")
            aabbs.construct(L, U)
            overlapping[:] = 0
            O = None
            if algorithm == algorithms[0]:
                O = sap.sort_and_sweep(parent, aabbs)
            if algorithm == algorithms[1]:
                bvh.build(aabbs, min, max)
                O = bvh.detect_overlaps(parent, aabbs)
            overlapping[O.flatten()] = 1
            vm.add_scalar_quantity(
                "Active simplices",
                overlapping,
                defined_on="cells",
                vminmax=(0, 1),
                enabled=True,
            )

            profiler.end_frame("Physics")
            t = t + 1

    ps.set_user_callback(callback)
    ps.show()
