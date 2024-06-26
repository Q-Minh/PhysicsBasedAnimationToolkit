import meshio
import argparse
import igl
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Volume mesh boundary extractor",
    )
    parser.add_argument("-i", "--input", help="Path to input mesh", type=str,
                        dest="input", required=True)
    parser.add_argument("-o", "--output", help="Path to output mesh", type=str,
                        dest="output", required=True)
    parser.add_argument("-r", "--remove-unreferenced", help="Remove unreferenced vertices", type=bool,
                        action=argparse.BooleanOptionalAction,
                        dest="remove_unreferenced", default=True)
    args = parser.parse_args()

    imesh = meshio.read(args.input)
    V, C = imesh.points, imesh.cells_dict["tetra"]
    F = igl.boundary_facets(C)
    F[:, :2] = np.roll(F[:, :2], shift=1, axis=1)
    V, F, I, J = igl.remove_unreferenced(V, F)
    omesh = meshio.Mesh(V, [("triangle", F)])
    meshio.write(args.output, omesh)
