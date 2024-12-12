# Mesh Graph Coloring Tool

A command-line and GUI application for performing graph coloring on meshes and visualizing the results.

## Description

This tool reads a mesh file, constructs the primal and dual graphs of the mesh, applies graph coloring algorithms, and visualizes the coloring on both vertices and cells using Polyscope.

- **Primal Graph**: Nodes represent mesh vertices; edges connect vertices that share a cell.
- **Dual Graph**: Nodes represent mesh cells (faces or elements); edges connect cells that share a vertex.

## Features

- **Supports surface and volume meshes**: Works with both triangle (surface) and tetrahedral (volume) meshes.
- **Interactive GUI**: Select coloring strategies and options in real-time.
- **Visualization**: Displays colored meshes highlighting the graph coloring results.
- **Multiple Coloring Strategies**: Choose from various graph coloring algorithms provided by NetworkX.

## Requirements

- Python 3.x
- [meshio](https://pypi.org/project/meshio/)
- [NetworkX](https://networkx.org/)
- [SciPy](https://www.scipy.org/)
- [NumPy](https://numpy.org/)
- [Polyscope](https://polyscope.run/)

## Installation

1. **Download the Script**

   Save the script to your local machine.

2. **Install Required Packages**

   Install the necessary Python packages using pip:

   ```bash
   pip install meshio networkx scipy numpy polyscope
   ```

## Usage

Run the script from the command line with the required input mesh file:

```bash
python mesh_graph_coloring.py -i <input_mesh>
```

### Command-Line Arguments

- `-i`, `--input`: **(Required)** Path to the input mesh file.

### Example

```bash
python mesh_graph_coloring.py -i mesh.obj
```

After running the script, a Polyscope window will open, allowing you to:

- Select a graph coloring strategy from a dropdown menu.
- Choose whether to use the interchange option in the coloring algorithm.
- Click the **"Color"** button to apply the coloring and visualize the results on the mesh.

## Supported Mesh Formats

The tool uses `meshio` for reading mesh files, supporting a variety of formats:

- `.obj`, `.off`, `.stl`, `.ply`, `.vtk`, `.vtu`, etc.

For a full list of supported formats, refer to the [meshio documentation](https://github.com/nschloe/meshio).

## How It Works

1. **Read the Input Mesh**

   The mesh is read using `meshio`, extracting vertices and elements (faces or cells).

2. **Construct Graphs**

   - **Primal Graph**: Created by connecting mesh vertices that share a common cell.
   - **Dual Graph**: Created by connecting mesh cells that share a common vertex.

3. **Initialize Visualization**

   Polyscope is initialized to render the mesh and set up the user interface.

4. **Interactive Graph Coloring**

   - **Select Strategy**: Choose a graph coloring strategy and whether to use interchange.
   - **Apply Coloring**: When the **"Color"** button is clicked, the tool applies the selected graph coloring algorithm to both the primal and dual graphs.
   - **Visualize Results**: The coloring is visualized on the mesh, with different colors representing different color assignments from the graph coloring algorithm.

## Graph Coloring Strategies

Available strategies from NetworkX include:

- `largest_first`
- `random_sequential`
- `smallest_last`
- `independent_set`
- `connected_sequential_bfs`
- `connected_sequential_dfs`
- `saturation_largest_first`
- `equitable`