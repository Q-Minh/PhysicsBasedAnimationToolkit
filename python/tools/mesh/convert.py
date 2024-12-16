import meshio
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Mesh format converter",
    )
    parser.add_argument("-i", "--input", help="Path to input mesh", type=str,
                        dest="input", required=True)
    parser.add_argument("-o", "--output", help="Path to output mesh", type=str,
                        dest="output", required=True)
    args = parser.parse_args()
    
    mesh = meshio.read(args.input)
    meshio.write(args.output, mesh)
