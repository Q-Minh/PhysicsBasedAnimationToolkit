import meshio
import argparse
import fast_simplification as fs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Triangle mesh simplification tool",
    )
    parser.add_argument("-i", "--input", help="Path to input mesh", type=str,
                        dest="input", required=True)
    parser.add_argument("-o", "--output", help="Path to output mesh", type=str,
                        dest="output", required=True)
    parser.add_argument("-t", "--target-reduction", help="Target reduction in mesh size in percentage",
                        type=float, dest="target_reduction", default="0.9")
    parser.add_argument("-a", "--aggressiveness",
                        help="High value means sacrifice quality for speed, low value means sacrifice speed for quality",
                        type=int, dest="aggressiveness", default=5)
    args = parser.parse_args()

    imesh = meshio.read(args.input)
    V, F = imesh.points, imesh.cells_dict["triangle"]
    V, F = fs.simplify(
        V, F, target_reduction=args.target_reduction, agg=args.aggressiveness)
    omesh = meshio.Mesh(V, [("triangle", F)])
    meshio.write(args.output, omesh)
