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
        "-i", "--input", help="Path to input mesh", dest="input", required=True
    )
    args = parser.parse_args()

    imesh = meshio.read(args.input)
    V, T = imesh.points.astype(np.float32), imesh.cells_dict["tetra"].astype(np.int32)
    F = igl.boundary_facets(T)
    F[:, :2] = np.roll(F[:, :2], shift=1, axis=1)
    profiler = pbat.profiling.Profiler()

    # Initial mesh quantities
    nverts = V.shape[0]
    ntets = T.shape[0]
    nfaces = F.shape[0]
    BV = np.hstack((np.zeros(nverts, dtype=np.int32), np.ones(nverts, dtype=np.int32)))

    # Duplicate input mesh into 2 separate meshes
    V = np.vstack((V.copy(), V.copy()))
    T = np.vstack((T.copy(), T + nverts))
    F = np.vstack((F.copy(), F + nverts))
    vinds = np.unique(F)
    cd = pbat.gpu.contact.VertexTriangleMixedCcdDcd(BV, vinds, F.T)
    XTG = pbat.gpu.common.Buffer(V.T)
    XG = pbat.gpu.common.Buffer(V.T)

    # Setup animation
    height = 3.5
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
    sm = ps.register_surface_mesh("Boundary Mesh", V, F)

    def callback():
        global dhat
        global t, speed, animate, export

        changed, speed = imgui.InputFloat("Speed", speed, format="%.4f")
        changed, export = imgui.Checkbox("Export", export)
        changed, animate = imgui.Checkbox("Animate", animate)
        step = imgui.Button("step")

        if animate or step:
            if export:
                ps.screenshot(f"{args.output}/{t}.png")

            profiler.begin_frame("Physics")
            if V[vimax, -1] >= zmax:
                direction[0] = -1
            if V[vimin, -1] <= zmin:
                direction[0] = 1
            if V[nverts + vimax, -1] >= zmax:
                direction[1] = -1
            if V[nverts + vimin, -1] <= zmin:
                direction[1] = 1
            V[:nverts, -1] = V[:nverts, -1] + direction[0] * speed
            V[nverts:, -1] = V[nverts:, -1] + direction[1] * speed
            sm.update_vertex_positions(V)

            XG.set(V.T)
            cd.initialize_active_set(XTG, XG, min, max)
            cd.update_active_set(XG)
            cd.finalize_active_set(XG)
            XTG.set(V.T)

            A = cd.active_set
            AV = np.zeros(V.shape[0])
            AF = np.zeros(F.shape[0])
            AV[vinds[A[0, :]]] = 1
            AF[A[1, :]] = 1

            sm.add_scalar_quantity(
                "Active Vertices", AV, enabled=True, cmap="coolwarm", vminmax=(0, 1)
            )
            sm.add_scalar_quantity(
                "Active Triangles",
                AF,
                defined_on="faces",
                enabled=False,
                cmap="coolwarm",
                vminmax=(0, 1),
            )

            profiler.end_frame("Physics")
            t = t + 1

    ps.set_user_callback(callback)
    ps.show()
