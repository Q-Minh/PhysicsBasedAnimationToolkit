# Volume Mesh Boundary Extractor

A command-line tool to extract the boundary surface mesh from a volumetric (tetrahedral) mesh.

## Description

The Volume Mesh Boundary Extractor reads a volumetric mesh file, extracts its boundary surface, and writes the surface mesh to a new file. This is useful for visualizing the outer surface of 3D meshes or preparing data for surface-based analyses.

## Features

- **Supports multiple mesh formats** via `meshio`.
- **Removes unreferenced vertices** to optimize the output mesh (optional).
- **Easy integration** into workflows that require surface extraction from volume meshes.

## Requirements

- Python 3.x
- [meshio](https://pypi.org/project/meshio/)
- [NumPy](https://numpy.org/)
- [libigl Python bindings (`pyigl`)](https://libigl.github.io/pyigl/)

## Installation

1. **Clone the repository or download the script**:

   ```bash
   git clone https://github.com/yourusername/volume-mesh-boundary-extractor.git
   ```

2. **Install the required Python packages**:

   ```bash
   pip install meshio numpy
   ```

3. **Install `pyigl`** (may require additional steps depending on your OS):

   ```bash
   pip install pyigl
   ```

   For detailed installation instructions, refer to the [libigl Python bindings documentation](https://libigl.github.io/pyigl/).

## Usage

Run the script from the command line with the necessary arguments:

```bash
python boundary.py -i <input_mesh> -o <output_mesh>
```

### Command-Line Arguments

- `-i`, `--input`: **(Required)** Path to the input volume mesh file.
- `-o`, `--output`: **(Required)** Path to save the output surface mesh file.
- `-r`, `--remove-unreferenced` / `--no-remove-unreferenced`: **(Optional)** Flag to remove unreferenced vertices from the output mesh. Enabled by default.

### Examples

**Extract boundary surface and remove unreferenced vertices (default behavior):**

```bash
python boundary.py -i mesh.msh -o surface.obj
```

**Extract boundary surface without removing unreferenced vertices:**

```bash
python boundary.py -i mesh.msh -o surface.obj --no-remove-unreferenced
```

## Supported Mesh Formats

The tool leverages `meshio` for reading and writing mesh files, supporting a wide range of formats:

- **Input Formats**: `.vtk`, `.vtu`, `.msh`, `.mesh`, etc.
- **Output Formats**: `.obj`, `.off`, `.stl`, `.ply`, etc.

For a complete list of supported formats, visit the [meshio documentation](https://github.com/nschloe/meshio).

## How It Works

1. **Read the Input Mesh**: Uses `meshio` to read the volumetric mesh file.
2. **Extract Boundary Facets**: Utilizes `libigl` to find the boundary faces of the tetrahedral mesh.
3. **Adjust Face Orientation**: Ensures the correct winding of the extracted surface faces.
4. **Remove Unreferenced Vertices**: Optionally cleans up the mesh by removing vertices not referenced by any face.
5. **Write the Output Mesh**: Saves the surface mesh to the specified file using `meshio`.