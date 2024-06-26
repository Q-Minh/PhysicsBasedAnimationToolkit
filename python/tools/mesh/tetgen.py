import tetgen as tg
import meshio
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Mesh viewer",
    )
    parser.add_argument("-i", "--input", help="Path to input triangle mesh", type=str,
                        dest="input", required=True)
    parser.add_argument("-o", "--output", help="Path to output triangle mesh", type=str,
                        dest="output", required=True)
    args = parser.parse_args()
    imesh = meshio.read(args.input)
    V, F = imesh.points, imesh.cells_dict["triangle"]
    mesher = tg.TetGen(V, F)
    Vtg, Ctg = mesher.tetrahedralize(
        order=1,
        steinerleft=V.shape[0],
        minratio=1.2,
        mindihedral=10.
    )
    omesh = meshio.Mesh(Vtg, [("tetra", Ctg)])
    meshio.write(args.output, omesh)
