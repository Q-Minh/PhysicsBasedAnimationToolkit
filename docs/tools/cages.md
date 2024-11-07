# Nested Mesh Cages Generation Tool

A command-line utility to generate nested cages around a given mesh, useful for applications like cage-based deformation, animation, or level-of-detail simplification.

## Description

This tool takes an input mesh and generates a cage (a simplified, enclosing mesh) that approximates the shape of the original mesh. The cage is generated using the `lazy_cage` algorithm from the `gpytoolbox` library. The resulting cage can be used for various geometric processing tasks where a simpler representation of the mesh is beneficial.

## Features

- **Customizable cage resolution**: Adjust the number of faces and grid size to control the cage's detail level.
- **Intersection-free cages**: Optionally enforce cages without self-intersections.
- **Supports multiple mesh formats** via `meshio`.
- **Command-line interface**: Easily integrate into scripts and workflows.

## Requirements

- Python 3.x
- [meshio](https://pypi.org/project/meshio/)
- [gpytoolbox](https://github.com/GeometryCollective/gpytoolbox)
- [NumPy](https://numpy.org/)

## Installation

1. **Clone the repository or download the script**:

   ```bash
   git clone https://github.com/yourusername/nested-mesh-cages-tool.git
   ```

2. **Install the required Python packages**:

   ```bash
   pip install meshio numpy
   ```

3. **Install `gpytoolbox`**:

   - If `gpytoolbox` is available via `pip`:

     ```bash
     pip install gpytoolbox
     ```

   - Otherwise, install from source:

     ```bash
     git clone https://github.com/GeometryCollective/gpytoolbox.git
     cd gpytoolbox
     python setup.py install
     ```

## Usage

Run the script from the command line with the necessary arguments:

```bash
python cages.py -i <input_mesh> -o <output_mesh>
```

### Command-Line Arguments

- `-i`, `--input`: **(Required)** Path to the input mesh file.
- `-o`, `--output`: **(Required)** Path to save the output cage mesh file.
- `--grid-size`: **(Optional)** Resolution of the signed distance grid used in the cage generation. Default is `50`.
- `--max-iters`: **(Optional)** Maximum number of iterations for the cage generation algorithm. Default is `10`.
- `--num-faces`: **(Optional)** Desired number of faces in the generated cage. Default is `100`.
- `--force-output`: **(Optional** Flag to accept cages that may have self-intersections. By default, the tool ensures the output cage is intersection-free.

### Examples

**Generate a cage with default parameters:**

```bash
python cages.py -i mesh.obj -o cage.obj
```

**Generate a cage with a higher resolution grid and more faces:**

```bash
python cages.py -i mesh.obj -o cage.obj --grid-size 100 --num-faces 200
```

**Generate a cage without enforcing intersection-free output:**

```bash
python cages.py -i mesh.obj -o cage.obj --force-output
```

## Supported Mesh Formats

The tool leverages `meshio` for reading and writing mesh files, supporting a wide range of formats:

- **Input Formats**: `.obj`, `.off`, `.stl`, `.ply`, `.vtk`, `.vtu`, etc.
- **Output Formats**: `.obj`, `.off`, `.stl`, `.ply`, etc.

For a complete list of supported formats, visit the [meshio documentation](https://github.com/nschloe/meshio).

## How It Works

1. **Read the Input Mesh**: Uses `meshio` to read the surface mesh file.
2. **Generate Cage**: Applies the `lazy_cage` algorithm from `gpytoolbox` to generate a cage around the mesh.
   - The algorithm uses a signed distance function and iterative optimization to create a simplified mesh that encloses the original mesh.
3. **Handle Output**:
   - If the cage generation is successful, the cage mesh is saved to the specified output file.
   - If it fails (e.g., due to convergence issues), the tool outputs an error message indicating the failure.
4. **Write the Output Mesh**: Saves the generated cage mesh to the specified file using `meshio`.
