import meshio
import argparse
from gpytoolbox.copyleft import lazy_cage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Nested Mesh Cages generation tool",
    )
    parser.add_argument("-i", "--input", help="Path to input mesh", type=str,
                        dest="input", required=True)
    parser.add_argument("-o", "--output", help="Path to output", type=str,
                        dest="output", required=True)
    parser.add_argument("--grid-size", help="Signed distance grid resolution", type=int,
                        dest="grid_size", default=50)
    parser.add_argument("--max-iters", help="Maximum number of iterations", type=int,
                        dest="max_iters", default=10)
    parser.add_argument("--num-faces", help="Desired number of cage faces", type=int,
                        dest="num_faces", default=100)
    parser.add_argument("--force-output", help="Accept intersecting output", action="store_true",
                        dest="force_output", default=False)
    args = parser.parse_args()

    imesh = meshio.read(args.input)
    V, F = imesh.points, imesh.cells_dict["triangle"]
    U, G = lazy_cage(V, F, grid_size=args.grid_size, max_iter=args.max_iters,
                     num_faces=args.num_faces, force_output=args.force_output)
    if U is None or G is None:
        print(f"Failed to generate cage for {args.input} with "
              f"grid size={args.grid_size}, "
              f"#max iterations={args.max_iters}, "
              f"#faces={args.num_faces}, "
              f"and intersection-free guarantee={not args.force_output}")
    else:
        omesh = meshio.Mesh(U, [("triangle", G)])
        meshio.write(args.output, omesh)
