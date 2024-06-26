import igl
import polyscope as ps
import numpy as np
import tetgen as tg
import meshio
import argparse

def subdivide(V, F, algorithm, k):
    if args.algorithm == "upsample":
        V, F = igl.upsample(V, F, k)
    if args.algorithm == "barycentric":
        for i in range(k):
            V, F = igl.false_barycentric_subdivision(V, F)
    if args.algorithm == "loop":
        V, F = igl.loop(V, F, k)
    return V, F

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Mesh refinement tool",
    )
    parser.add_argument("-i", "--input", help="Path to input mesh", type=str,
                        dest="input", required=True)
    parser.add_argument("-o", "--output", help="Path to output mesh", type=str,
                        dest="output", required=True)
    parser.add_argument("-m", "--mesh-type", help="tri | tet",
                        type=str, dest="mesh_type", required=True)
    parser.add_argument("-k", "--num-refinement-pass", help="Number of refinement passes", type=int,
                        dest="k")
    parser.add_argument("-a", "--algorithm", help="Triangle mesh subdivision algorithm used, one of upsample | barycentric | loop", type=str,
                        dest="algorithm", default="upsample")
    args = parser.parse_args()

    if args.mesh_type == "tri":
        V, F = igl.read_triangle_mesh(args.input)
        V, F = subdivide(V, F, args.algorithm, args.k)
        mesh = meshio.Mesh(V, [("triangle", F)])
        meshio.write(args.output, mesh)
    elif args.mesh_type == "tet":
        mesh = meshio.read(args.input)
        V, C = mesh.points, mesh.cells_dict["tetra"]
        vol = igl.volume(V, C)
        F = igl.boundary_facets(C)
        F[:, :2] = np.roll(F[:, :2], shift=1, axis=1)
        Vsub, Fsub = subdivide(V, F, args.algorithm, args.k)
        mesher = tg.TetGen(Vsub, Fsub)
        Vrefined, Crefined = mesher.tetrahedralize(
            order=1,
            steinerleft=int(2*Vsub.shape[0]),
            minratio=1.2,
            mindihedral=10.,
            maxvolume=vol.mean()/(4**args.k)
        )
        mesh = meshio.Mesh(Vrefined, [("tetra", Crefined)])
        meshio.write(args.output, mesh)
    else:
        print(f"Could not refine unknown mesh type {args.mesh_type}")

