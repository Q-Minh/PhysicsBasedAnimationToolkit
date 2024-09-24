import numpy as np
import meshio
import tetgen as tg
import argparse
import os
from stl import mesh as stl_mesh
import trimesh

def create_cube(size):
    """Create a cube mesh with a given size."""
    vertices = np.array([[0, 0, 0],
                         [size, 0, 0],
                         [size, size, 0],
                         [0, size, 0],
                         [0, 0, size],
                         [size, 0, size],
                         [size, size, size],
                         [0, size, size]])
    
    faces = np.array([[0, 1, 2],
                      [0, 2, 3],
                      [4, 5, 6],
                      [4, 6, 7],
                      [0, 1, 5],
                      [0, 5, 4],
                      [1, 2, 6],
                      [1, 6, 5],
                      [2, 3, 7],
                      [2, 7, 6],
                      [3, 0, 4],
                      [3, 4, 7]])
    
    return vertices, faces

def create_rectangle(length, width, height):
    """Create a rectangular prism mesh with given dimensions."""
    vertices = np.array([[0, 0, 0],
                         [length, 0, 0],
                         [length, width, 0],
                         [0, width, 0],
                         [0, 0, height],
                         [length, 0, height],
                         [length, width, height],
                         [0, width, height]])
    
    faces = np.array([[0, 1, 2],
                      [0, 2, 3],
                      [0, 1, 5],
                      [0, 5, 4],
                      [1, 2, 6],
                      [1, 6, 5],
                      [2, 3, 7],
                      [2, 7, 6],
                      [3, 0, 4],
                      [3, 4, 7],
                      [4, 5, 6],
                      [4, 6, 7]])
    
    return vertices, faces

def create_sphere(radius, resolution):
    """Create a sphere mesh with a given radius and resolution."""
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi = np.linspace(0, np.pi, resolution)
    
    x = radius * np.outer(np.sin(phi), np.cos(theta))
    y = radius * np.outer(np.sin(phi), np.sin(theta))
    z = radius * np.outer(np.cos(phi), np.ones_like(theta))

    vertices = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            faces.append([i * resolution + j, i * resolution + (j + 1), (i + 1) * resolution + j])
            faces.append([(i + 1) * resolution + j, i * resolution + (j + 1), (i + 1) * resolution + (j + 1)])
    
    faces = np.array(faces)
    return vertices, faces



def create_tetrahedron():
    """Create a tetrahedron mesh."""
    vertices = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [0.5, np.sqrt(3) / 2, 0],
                         [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3]])
    
    faces = np.array([[0, 1, 2],
                      [0, 1, 3],
                      [1, 2, 3],
                      [2, 0, 3]])
    
    return vertices, faces

def create_cylinder(radius, height, resolution):
    """Create a closed cylinder mesh with given radius, height, and resolution."""
    # Generate vertices for the top and bottom circles
    theta = np.linspace(0, 2 * np.pi, resolution)
    
    # Bottom circle
    bottom_circle = np.array([[radius * np.cos(t), radius * np.sin(t), 0] for t in theta])
    # Top circle
    top_circle = np.array([[radius * np.cos(t), radius * np.sin(t), height] for t in theta])
    
    # Center points for top and bottom
    center_bottom = np.array([0, 0, 0])
    center_top = np.array([0, 0, height])

    # Combine vertices
    vertices = np.vstack([bottom_circle, top_circle, center_bottom, center_top])

    # Create side faces
    faces = []
    for i in range(resolution):
        next_i = (i + 1) % resolution
        
        # Side faces
        faces.append([i, next_i, resolution + next_i])  # Bottom triangle
        faces.append([i, resolution + next_i, resolution + i])  # Top triangle

        # Bottom cap
        faces.append([i, next_i, 2 * resolution])  # Bottom center point is the last vertex

        # Top cap
        faces.append([resolution + i, resolution + next_i, 2 * resolution + 1])  # Top center point

    faces = np.array(faces)
    return vertices, faces

def save_stl(vertices, faces, filename):
    """Save the mesh to an STL file."""
    cube = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[f[j],:]
    cube.save(filename)

def generate_tetrahedral_mesh(vertices, faces):
    """Generate a tetrahedral mesh from the given surface mesh, with mesh repair using trimesh."""
    
    # Create a trimesh object
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Repair the mesh
    trimesh_mesh.fix_normals()
    trimesh_mesh.remove_degenerate_faces()
    trimesh_mesh.fill_holes()
    trimesh_mesh.remove_duplicate_faces()
    trimesh_mesh.remove_infinite_values()
    trimesh_mesh.remove_unreferenced_vertices()

    if not trimesh_mesh.is_watertight:
        raise RuntimeError("The mesh is not watertight even after repair. Cannot proceed with tetrahedralization.")
    
    # Convert the repaired trimesh object back to vertices and faces
    vertices_repaired = trimesh_mesh.vertices
    faces_repaired = trimesh_mesh.faces

    # Use TetGen for tetrahedralization
    mesher = tg.TetGen(vertices_repaired, faces_repaired)
    Vtg, Ctg = mesher.tetrahedralize(
        order=1,
        steinerleft=vertices.shape[0],
        minratio=1.2,
        mindihedral=10.
    )
    
    return Vtg, Ctg

def main():
    parser = argparse.ArgumentParser(description="Generate and save a mesh of various shapes in STL or mesh format.")
    parser.add_argument("-t", "--type", choices=['cube', 'rectangle', 'sphere', 'tetrahedron', 'cylinder'], required=True, help="Type of the mesh to generate.")
    parser.add_argument("-s", "--size", type=float, default=1.0, help="Size for cube or length for rectangular prism. Default is 1.0.")
    parser.add_argument("-w", "--width", type=float, default=1.0, help="Width of the rectangular prism (required if type is 'rectangle'). Default is 1.0.")
    parser.add_argument("--height", type=float, default=1.0, help="Height of the rectangular prism or cylinder. Default is 1.0.")
    parser.add_argument("--radius", type=float, default=1.0, help="Radius of the sphere or cylinder. Default is 1.0.")
    parser.add_argument("--resolution", type=int, default=20, help="Resolution of the sphere or cylinder. Default is 20.")
    parser.add_argument("-f", "--format", choices=['stl', 'mesh'], required=True, help="Output file format: 'stl' or 'mesh'.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file name.")
    
    args = parser.parse_args()
    
    if args.type == 'cube':
        vertices, faces = create_cube(args.size)
    elif args.type == 'rectangle':
        vertices, faces = create_rectangle(args.size, args.width, args.height)
    elif args.type == 'sphere':
        vertices, faces = create_sphere(args.radius, args.resolution)
    elif args.type == 'tetrahedron':
        vertices, faces = create_tetrahedron()
    elif args.type == 'cylinder':
        vertices, faces = create_cylinder(args.radius, args.height, args.resolution)
    
    if args.format == 'stl':
        save_stl(vertices, faces, args.output)
    elif args.format == 'mesh':
        Vtg, Ctg = generate_tetrahedral_mesh(vertices, faces)
        omesh = meshio.Mesh(Vtg, [("tetra", Ctg)])
        meshio.write(args.output, omesh)
    
    print(f"Mesh saved to {args.output}")

if __name__ == "__main__":
    main()
