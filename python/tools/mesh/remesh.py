import meshio
import argparse
import gpytoolbox as gpyt
import igl
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Triangle mesh remeshing tool",
    )
    parser.add_argument("-i", "--input", help="Path to input mesh", type=str,
                        dest="input", required=True)
    parser.add_argument("-o", "--output", help="Path to output mesh", type=str,
                        dest="output", required=True)
    parser.add_argument("-p", "--project", help="Project remeshed vertices onto original mesh", type=bool,
                        action=argparse.BooleanOptionalAction,
                        dest="project", default=True)
    parser.add_argument("-k", "--iterations",
                        help="Number of remeshing iterations",
                        type=int, dest="iterations", default=10)
    parser.add_argument("-f", "--feature-dihedral-angle",
                        help="Minimum dihedral angle for an edge to be considered sharp",
                        type=float, dest="feature_dihedral", default=0.1)
    args = parser.parse_args()

    imesh = meshio.read(args.input)
    V, F = imesh.points, imesh.cells_dict["triangle"]
    SE, E, uE, Emap, uE2E, sharp = igl.sharp_edges(V, F, args.feature_dihedral)
    uEsharp = uE[sharp, :]
    feature_vertices = np.unique(uEsharp)
    V, F = gpyt.remesh_botsch(V, F, i=args.iterations,
                              project=args.project, feature=feature_vertices)
    omesh = meshio.Mesh(V, [("triangle", F)])
    meshio.write(args.output, omesh)
