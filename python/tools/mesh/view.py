import polyscope as ps
import meshio
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Mesh viewer",
    )
    parser.add_argument("-i", "--input", help="Paths to input mesh", nargs="+",
                        dest="inputs", required=True)
    args = parser.parse_args()
    
    ps.init()
    for input in args.inputs:
        mesh = meshio.read(input)
        if "tetra" in mesh.cells_dict:
            ps.register_volume_mesh(input, mesh.points, mesh.cells_dict["tetra"])
        elif "triangle" in mesh.cells_dict:
            ps.register_surface_mesh(input, mesh.points, mesh.cells_dict["triangle"])
        else:
            print(f"Input mesh {input} not supported")
    ps.show()