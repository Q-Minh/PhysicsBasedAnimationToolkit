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
    parser.add_argument("-t", "--translation", help="Vertical translation", type=float,
                        dest="translation", default=0.1)
    args = parser.parse_args()

    imesh = meshio.read(args.input)
    V, C = imesh.points.astype(
        np.float32), imesh.cells_dict["tetra"].astype(np.int32)

    # Duplicate input mesh into 2 separate meshes
    ntets = C.shape[0]
    nverts = V.shape[0]
    T = np.vstack([np.copy(C), np.copy(C) + nverts])
    V = np.vstack([V.copy(), V.copy()])
    profiler = pbat.profiling.Profiler()
    S = pbat.gpu.geometry.Simplices(T.T)
    lbvh = pbat.gpu.geometry.LinearBvh(T.shape[0], 60*T.shape[0])

    # Setup animation
    height = 4
    vimin = V[:nverts, -1].argmin()
    vimax = V[:nverts, -1].argmax()
    zmin = V[vimin, -1]
    zextent = V[vimax, -1] - zmin
    zmax = height * zextent + zmin
    min, max = np.min(V, axis=0), np.max(V, axis=0)
    min[-1] = zmin
    max[-1] = zmax
    V[nverts:, -1] += (height - 2) * zextent
    P = pbat.gpu.geometry.Points(V.T)
    direction = [1, -1]

    # Setup GUI
    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Linear BVH Broad Phase")
    ps.init()

    speed = 0.01
    animate = False
    dhat = 0.
    o = 0
    O = np.zeros((2,0), dtype=np.int32)
    overlapping = np.zeros(T.shape[0])
    vm = ps.register_volume_mesh(f"Mesh", V, T)

    def callback():
        global dhat
        global speed
        global animate
        global o, overlapping, O

        changed, dhat = imgui.InputFloat(
            "Box expansion", dhat, format="%.4f")
        changed, speed = imgui.InputFloat(
            "Speed", speed, format="%.4f")
        changed, animate = imgui.Checkbox("animate", animate)
        changed, o = imgui.InputInt("o", o)
        step = imgui.Button("step")

        if animate or step:
            profiler.begin_frame("Physics")
            if V[vimax, -1] >= zmax:
                direction[0] = -1
            if V[vimin, -1] <= zmin:
                direction[0] = 1
            if V[nverts+vimax, -1] >= zmax:
                direction[1] = -1
            if V[nverts+vimin, -1] <= zmin:
                direction[1] = 1
            V[:nverts, -1] += direction[0]*speed
            V[nverts:, -1] += direction[1]*speed
            vm.update_vertex_positions(V)
            P.V = V.T
            min, max = np.min(V, axis=0), np.max(V, axis=0)
            lbvh.build(P, S, min, max, expansion=dhat)
            O = lbvh.detect_self_overlaps(S)
            overlapping[:] = 0
            # overlapping[O.flatten()] = 1
            # simplex = lbvh.simplex
            # n = simplex.shape[0]
            # b, e = lbvh.b[n-1:, :], lbvh.e[n-1:, :]
            # b, e = b[simplex, :], e[simplex, :]
            # intersects = (e[O[0, :], 0] >= b[O[1, :], 0]) & (e[O[0, :], 1] >= b[O[1, :], 1]) & (e[O[0, :], 2] >= b[O[1, :], 2]) & (
            #     b[O[0, :], 0] <= e[O[1, :], 0]) & (b[O[0, :], 1] <= e[O[1, :], 1]) & (b[O[0, :], 2] <= e[O[1, :], 2])
            # overlapping[O[:, np.nonzero(intersects)].flatten()] = 1
            # vm.add_scalar_quantity(
            #     "Active simplices", overlapping, defined_on="cells", vminmax=(0, 1), enabled=True)
            profiler.end_frame("Physics")
            
        if o < O.shape[1]:
            ps.register_volume_mesh("Overlapping", V, T[O[:,o],:])
            overlapping[O[:,o]] = 1

    ps.set_user_callback(callback)
    ps.show()
