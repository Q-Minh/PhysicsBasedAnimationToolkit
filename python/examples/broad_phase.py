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
        "-i", "--input", help="Path to input mesh", dest="input", required=True)
    parser.add_argument(
        "-o", "--output", help="Path to output", dest="output", required=False, default=".")
    parser.add_argument("-t", "--translation", help="Vertical translation", type=float,
                        dest="translation", default=0.1)
    args = parser.parse_args()

    imesh = meshio.read(args.input)
    V, C = imesh.points.astype(
        np.float64), imesh.cells_dict["tetra"].astype(np.int64)

    # Duplicate input mesh into 2 separate meshes
    T = [np.copy(C), np.copy(C)]
    S = [
        pbat.gpu.geometry.Simplices(T[0].T),
        pbat.gpu.geometry.Simplices((T[1] + V.shape[0]).T)
    ]
    sap = pbat.gpu.geometry.SweepAndPrune(2*T[0].shape[0], 48*T[0].shape[0])
    query = pbat.gpu.geometry.BvhQuery(T[0].shape[0], 48*T[0].shape[0], 0)
    bvh = pbat.gpu.geometry.Bvh(T[1].shape[0], 0)
    profiler = pbat.profiling.Profiler()

    # Setup animation
    height = 3
    vimin = V[:, -1].argmin()
    vimax = V[:, -1].argmax()
    zmin = V[vimin, -1]
    zextent = V[vimax, -1] - zmin
    zmax = (height - 1) * zextent + zmin
    V = [np.copy(V), np.copy(V)]
    V[-1][:, -1] = V[-1][:, -1] + (height - 2) * zextent
    P = pbat.gpu.geometry.Points(np.vstack(V).T)
    direction = [1, -1]
    min, max = np.min(P.V, axis=1), np.max(P.V, axis=1)
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
    dhat = 0.
    algorithms = [
        "Sweep and Prune",
        "Bounding Volume Hierarchy",
    ]
    algorithm = algorithms[0]
    export = False
    overlapping = [np.zeros(T[0].shape[0]), np.zeros(T[1].shape[0])]
    vm = [ps.register_volume_mesh(
        f"Mesh 0", V[0], T[0]), ps.register_volume_mesh(f"Mesh 1", V[1], T[1])]

    def callback():
        global dhat
        global t, speed, animate, algorithms, algorithm, export

        changed = imgui.BeginCombo("Algorithm", algorithm)
        if changed:
            for i in range(len(algorithms)):
                _, selected = imgui.Selectable(
                    algorithms[i], algorithm == algorithms[i])
                if selected:
                    algorithm = algorithms[i]
            imgui.EndCombo()
        changed, dhat = imgui.InputFloat(
            "Box expansion", dhat, format="%.4f")
        changed, speed = imgui.InputFloat(
            "Speed", speed, format="%.4f")
        changed, export = imgui.Checkbox("Export", export)
        changed, animate = imgui.Checkbox("animate", animate)
        step = imgui.Button("step")

        if animate or step:
            if export:
                ps.screenshot(f"{args.output}/{t}.png")

            profiler.begin_frame("Physics")
            for i in range(len(V)):
                if V[i][vimax, -1] >= zmax:
                    direction[i] = -1
                if V[i][vimin, -1] <= zmin:
                    direction[i] = 1
                V[i][:, -1] = V[i][:, -1] + direction[i] * speed
                vm[i].update_vertex_positions(V[i])
            P.V = np.vstack(V).T
            if algorithm == algorithms[0]:
                O = sap.sort_and_sweep(
                    P, S[0], S[1], dhat)
            if algorithm == algorithms[1]:
                query.build(P, S[0], min, max, expansion=dhat)
                bvh.build(P, S[1], min, max, expansion=dhat)
                O = query.detect_overlaps(P, S[0], S[1], bvh)
            for i in range(len(overlapping)):
                overlapping[i][:] = 0
                overlapping[i][O[i, :]] = 1
                vm[i].add_scalar_quantity(
                    "Active simplices", overlapping[i], defined_on="cells", vminmax=(0, 1), enabled=True)
            profiler.end_frame("Physics")
            t = t + 1

    ps.set_user_callback(callback)
    ps.show()
