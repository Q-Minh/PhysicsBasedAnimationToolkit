import pbatoolkit as pbat
import argparse
import meshio
import numpy as np
import polyscope as ps
import polyscope.imgui as imgui
import igl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="3D vertex-triangle contact detection",
    )
    parser.add_argument(
        "-i", "--input", help="Path to input mesh", dest="input", required=True)
    args = parser.parse_args()

    imesh = meshio.read(args.input)
    V, T = imesh.points.astype(
        np.float32), imesh.cells_dict["tetra"].astype(np.int32)
    F = igl.boundary_facets(T)
    F[:, :2] = np.roll(F[:, :2], shift=1, axis=1)
    profiler = pbat.profiling.Profiler()

    # Initial mesh quantities
    nverts = V.shape[0]
    ntets = T.shape[0]
    nfaces = F.shape[0]
    BV = np.hstack((np.zeros(nverts, dtype=np.int32),
                   np.ones(nverts, dtype=np.int32)))
    BF = np.hstack((np.zeros(nfaces, dtype=np.int32),
                   np.ones(nfaces, dtype=np.int32)))

    # Duplicate input mesh into 2 separate meshes
    V = np.vstack((V.copy(), V.copy()))
    T = np.vstack((T.copy(), T + nverts))
    F = np.vstack((F.copy(), F + nverts))
    vinds = np.linspace(
        0, V.shape[0]-1, num=V.shape[0], dtype=np.int32)[np.newaxis, :]
    SV = pbat.gpu.geometry.Simplices(
        vinds)
    ST = pbat.gpu.geometry.Simplices(T.T)
    SF = pbat.gpu.geometry.Simplices(F.T)
    Vquery = pbat.gpu.geometry.BvhQuery(
        V.shape[0], 24*T.shape[0], 8*F.shape[0])
    Tbvh = pbat.gpu.geometry.Bvh(T.shape[0], 0)
    Fbvh = pbat.gpu.geometry.Bvh(F.shape[0], 0)
    BV = pbat.gpu.geometry.Bodies(BV)
    BF = pbat.gpu.geometry.Bodies(BF)

    # Setup animation
    height = 3
    vimin = V[:nverts, -1].argmin()
    vimax = V[:nverts, -1].argmax()
    zmin = V[vimin, -1]
    zextent = V[vimax, -1] - zmin
    zmax = (height - 1) * zextent + zmin
    V[nverts:, -1] = V[nverts:, -1] + (height - 2) * zextent
    direction = [1, -1]
    min, max = np.min(V, axis=0), np.max(V, axis=0)
    min[-1] = zmin
    max[-1] = zmax
    P = pbat.gpu.geometry.Points(V.T)

    # Setup GUI
    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Vertex-Triangle Contact Detection")
    ps.init()

    t = 0
    speed = 0.01
    animate = False
    dhat = zmax - zmin
    export = False
    show_nn_pairs = False
    show_all_nn_pairs = False
    nearest = np.zeros(F.shape[0])
    N = np.zeros((2, 0), dtype=np.int32)
    n = 0
    voverlaps = np.zeros(V.shape[0])
    toverlaps = np.zeros(T.shape[0])
    sm = ps.register_surface_mesh("Boundary Mesh", V, F)
    vm = ps.register_volume_mesh("Mesh", V, T, enabled=False)

    def callback():
        global dhat
        global t, speed, animate, N, n, export, show_nn_pairs, show_all_nn_pairs

        changed, dhat = imgui.InputFloat(
            "NN search radius", dhat, format="%.4f")
        changed, speed = imgui.InputFloat(
            "Speed", speed, format="%.4f")
        changed, n = imgui.InputInt("NN pair", n)
        changed, export = imgui.Checkbox("Export", export)
        changed, animate = imgui.Checkbox("Animate", animate)
        changed, show_nn_pairs = imgui.Checkbox(
            "Display neighbours", show_nn_pairs)
        changed, show_all_nn_pairs = imgui.Checkbox(
            "Display all neighbours", show_all_nn_pairs)
        step = imgui.Button("step")
        imgui.Text(f"# nearest neighbor pairs={N.shape[1]}")

        if animate or step:
            if export:
                ps.screenshot(f"{args.output}/{t}.png")

            profiler.begin_frame("Physics")
            if V[vimax, -1] >= zmax:
                direction[0] = -1
            if V[vimin, -1] <= zmin:
                direction[0] = 1
            if V[nverts+vimax, -1] >= zmax:
                direction[1] = -1
            if V[nverts+vimin, -1] <= zmin:
                direction[1] = 1
            V[:nverts, -1] = V[:nverts, -1] + direction[0] * speed
            V[nverts:, -1] = V[nverts:, -1] + direction[1] * speed
            sm.update_vertex_positions(V)
            vm.update_vertex_positions(V)
            P.V = V.T
            Vquery.build(P, SV, min, max)
            Tbvh.build(P, ST, min, max)
            Fbvh.build(P, SF, min, max)
            O = Vquery.detect_overlaps(P, SV, ST, Tbvh)
            voverlaps[:] = 0.
            toverlaps[:] = 0.
            nearest[:] = 0.
            if O.shape[1] > 0:
                voverlaps[O[0, :]] = 1.
                toverlaps[O[1, :]] = 1.

            sm.add_scalar_quantity(
                "Active vertices", voverlaps, vminmax=(0, 1), enabled=True, cmap="coolwarm")
            vm.add_scalar_quantity("Active vertices", voverlaps, vminmax=(
                0, 1), enabled=True, cmap="coolwarm")
            vm.add_scalar_quantity("Active tetrahedra", toverlaps, defined_on="cells", vminmax=(
                0, 1), enabled=False, cmap="coolwarm")

            N = Vquery.detect_contact_pairs(P, SV, SF, BV, BF, Fbvh, dhat)
            if N.shape[1] > 0:
                nearest[N[1, :]] = 1.
            sm.add_scalar_quantity(
                "Active triangles", nearest, defined_on="faces", vminmax=(0, 1), enabled=True, cmap="coolwarm")
            profiler.end_frame("Physics")
            t = t + 1

        if show_nn_pairs and n < N.shape[1]:
            ps.register_point_cloud(
                "Active vertex", V[N[0, n], :][np.newaxis, :])
            ps.register_surface_mesh(
                "Nearest triangle", V, F[N[1, n], :][np.newaxis, :])
        if show_all_nn_pairs and N.shape[1] > 0:
            ps.register_point_cloud("Active vertices", V[N[0, :], :])
            ps.register_surface_mesh(
                "Active triangles", V, F[N[1, :], :])

    ps.set_user_callback(callback)
    ps.show()
