import meshio
import argparse
import pymeshfix as pmf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Triangle mesh repair tool",
    )
    parser.add_argument("-i", "--input", help="Path to input mesh", type=str,
                        dest="input", required=True)
    parser.add_argument("-o", "--output", help="Path to output mesh", type=str,
                        dest="output", required=True)
    args = parser.parse_args()

    imesh = meshio.read(args.input)
    V, F = imesh.points, imesh.cells_dict["triangle"]
    V, F = pmf.clean_from_arrays(V, F)
    omesh = meshio.Mesh(V, [("triangle", F)])
    meshio.write(args.output, omesh)
