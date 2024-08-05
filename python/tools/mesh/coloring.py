import networkx as nx
import scipy as sp
import numpy as np
import meshio
import argparse
import polyscope as ps
import polyscope.imgui as psim
import math


def color_dict_to_array(Cdict, n):
    C = np.zeros(n)
    keys = [key for key in Cdict.keys()]
    values = [value for value in Cdict.values()]
    C[keys] = values
    return C


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Graph coloring on meshes",
    )
    parser.add_argument("-i", "--input", help="Path to input mesh",
                        dest="input", required=True)
    args = parser.parse_args()

    imesh = meshio.read(args.input)
    V = imesh.points
    if "triangle" in imesh.cells_dict:
        C = imesh.cells_dict["triangle"]
    if "tetra" in imesh.cells_dict:
        C = imesh.cells_dict["tetra"]

    row = np.repeat(range(C.shape[0]), C.shape[1])
    col = C.flatten()
    data = np.ones(math.prod(C.shape))
    G = sp.sparse.coo_array((data, (row, col)), shape=(
        C.shape[0], V.shape[0])).asformat("csr")
    Gdual = nx.Graph(G @ G.T)
    Gprimal = nx.Graph(G.T @ G)

    ps.set_program_name("Mesh Graph Coloring")
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.init()

    if C.shape[1] == 3:
        m = ps.register_surface_mesh("Mesh", V, C)
    if C.shape[1] == 4:
        m = ps.register_volume_mesh("Mesh", V, C)
    defined_on = "faces" if C.shape[1] == 3 else "cells"
    m.set_transparency(0.6)
    m.set_edge_width(1.)

    strategies = [
        "largest_first",
        "random_sequential",
        "smallest_last",
        "independent_set",
        "connected_sequential_bfs",
        "connected_sequential_dfs",
        "saturation_largest_first",
        "equitable"
    ]
    strategy = strategies[0]
    interchange = False

    def callback():
        global strategies, strategy, interchange
        changed = psim.BeginCombo("Strategy", strategy)
        if changed:
            for i in range(len(strategies)):
                _, selected = psim.Selectable(
                    strategies[i], strategy == strategies[i])
                if selected:
                    strategy = strategies[i]
            psim.EndCombo()

        _, interchange = psim.Checkbox("Use interchange", interchange)
        if psim.Button("Color"):
            if strategy == "equitable":
                ddual = max([d for n, d in Gdual.degree])
                dprimal = max([d for n, d in Gprimal.degree])
                GCdual = nx.equitable_color(Gdual, num_colors=ddual+1)
                GCdual = color_dict_to_array(GCdual, C.shape[0])
                GCprimal = nx.equitable_color(Gprimal, num_colors=dprimal+1)
                GCprimal = color_dict_to_array(GCprimal, V.shape[0])
            else:
                GCdual = nx.greedy_color(Gdual, strategy=strategy)
                GCdual = color_dict_to_array(GCdual, C.shape[0])
                GCprimal = nx.greedy_color(Gprimal, strategy=strategy)
                GCprimal = color_dict_to_array(GCprimal, V.shape[0])

            m.add_scalar_quantity(
                f"Cell colors", GCdual, defined_on=defined_on, cmap="jet", enabled=True)
            m.add_scalar_quantity(f"Vertex colors", GCprimal,
                                  cmap="jet", enabled=True)
    ps.set_user_callback(callback)
    ps.show()
