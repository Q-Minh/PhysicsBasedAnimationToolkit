import meshio
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Generate C++ code using Eigen to construct input mesh",
    )
    parser.add_argument("-i", "--input", help="Path to input mesh", type=str,
                        dest="input", required=True)
    parser.add_argument("-o", "--output", help="Path to output file", type=str,
                        dest="output", required=True)
    args = parser.parse_args()
    
    mesh = meshio.read(args.input)
    V, C = mesh.points, mesh.cells_dict["tetra"]
    with open(args.output, "w") as file:
        file.write(f"MatrixX V({V.shape[1]},{V.shape[0]});\n")
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                file.write(f"V({j},{i})={V[i,j]};\n")
        file.write(f"IndexMatrixX C({C.shape[1]},{C.shape[0]});\n")
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                file.write(f"C({j},{i})={C[i,j]};\n")
