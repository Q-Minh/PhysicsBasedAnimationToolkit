import pbatoolkit as pbat
import meshio
import polyscope as ps
import polyscope.imgui as imgui
import argparse
import numpy as np

if __name__ == "__main__":
    # Load input mesh
    parser = argparse.ArgumentParser(
        prog="3D point-triangle nearest neighbor search",
    )
    parser.add_argument(
        "-i", "--input", help="Path to input triangle mesh", dest="input", required=True
    )
    args = parser.parse_args()
    imesh = meshio.read(args.input)
    V, F = imesh.points.astype(np.float32), imesh.cells_dict["triangle"].astype(
        np.int32
    )

    # Compute BVH over V,F
    aabbs = pbat.gpu.geometry.Aabb(3, F.shape[0])
    aabbs.construct(V.T, F.T)
    bvh = pbat.gpu.geometry.Bvh(F.shape[0], 0)
    min, max = np.min(V, axis=0), np.max(V, axis=0)
    bvh.build(aabbs, min, max)

    # GPU quantities
    X = np.zeros((3, 2), dtype=np.float32, order="F")
    XG = pbat.gpu.common.Buffer(X)
    VG = pbat.gpu.common.Buffer(V.T)
    FG = pbat.gpu.common.Buffer(F.T)

    # Setup GUI
    ps.set_verbosity(0)
    ps.set_program_name("Point-Triangle NN Search")
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_front_dir("neg_y_front")
    ps.init()

    nnmap = np.zeros(F.shape[0], dtype=np.float32)
    pc = ps.register_point_cloud("Query Point", X.T)
    tm = ps.register_surface_mesh("mesh", V, F)

    def callback():
        global X
        changed = [None] * 3
        changed[0], X[0, :] = imgui.SliderFloat("X", X[0, 0], v_min=-5, v_max=5)
        changed[1], X[1, :] = imgui.SliderFloat("Y", X[1, 0], v_min=-5, v_max=5)
        changed[2], X[2, :] = imgui.SliderFloat("Z", X[2, 0], v_min=-5, v_max=5)
        pc.update_point_positions(X.T)
        nn = bvh.point_triangle_nearest_neighbours(aabbs, XG, VG, FG)
        nnmap[:] = 0
        nnmap[nn] = 1
        tm.add_scalar_quantity(
            "Nearest Neighbour", nnmap, defined_on="faces", enabled=True
        )

    ps.set_user_callback(callback)
    ps.show()
