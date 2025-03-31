import pbatoolkit as pbat
import argparse
import meshio
import numpy as np
import polyscope as ps
import polyscope.imgui as imgui

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="3D multibody mesh CCD and DCD contact detection algorithm",
    )
    parser.add_argument(
        "-i", "--input", help="Path to input mesh", dest="input", required=True
    )
    args = parser.parse_args()

    # Load input mesh
    imesh = meshio.read(args.input)
    V, T = imesh.points.astype(np.float64), imesh.cells_dict["tet"].astype(np.int64)
