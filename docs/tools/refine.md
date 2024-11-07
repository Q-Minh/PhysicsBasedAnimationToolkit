# Mesh Refinement Tool

A command-line utility to refine 2D and 3D meshes using various subdivision algorithms.

## Description

This tool reads an input mesh and performs refinement by subdividing its elements (triangles or tetrahedra) using specified algorithms. It supports surface meshes (triangles) and volume meshes (tetrahedra).

## Features

- **Supports Triangle and Tetrahedral Meshes**
- **Multiple Subdivision Algorithms**: Choose from upsample, barycentric, or Loop subdivision for triangle meshes.
- **Customizable Refinement Levels**: Control the number of refinement passes.

## Requirements

- Python 3.x
- `numpy`
- `meshio`
- `pyigl` (Python bindings for libigl)
- `tetgen` Python wrapper

## Installation

Install the required Python packages:

```bash
pip install numpy meshio pyigl tetgen
```

Note: Installation of `pyigl` and `tetgen` may require additional steps. Refer to their respective documentation.

## Usage

Run the script from the command line with the necessary arguments:

```bash
python refine.py -i input_mesh.ext -o output_mesh.ext -m tri -k 2 -a loop
```

### Arguments

- `-i`, `--input`: **(Required)** Path to the input mesh file.
- `-o`, `--output`: **(Required)** Path to save the refined mesh file.
- `-m`, `--mesh-type`: **(Required)** Type of mesh: `tri` for triangle meshes or `tet` for tetrahedral meshes.
- `-k`, `--num-refinement-pass`: **(Optional)** Number of refinement passes (default: `1`).
- `-a`, `--algorithm`: **(Optional)** Subdivision algorithm for triangle meshes. Options are `upsample`, `barycentric`, or `loop` (default: `upsample`).

### Examples

**Refine a triangle mesh using Loop subdivision:**

```bash
python refine.py -i surface.obj -o refined_surface.obj -m tri -k 2 -a loop
```

**Refine a tetrahedral mesh:**

```bash
python refine.py -i volume.vtk -o refined_volume.vtk -m tet -k 1
```

## Supported Mesh Formats

The tool uses `meshio` for reading and writing mesh files, supporting formats like `.obj`, `.stl`, `.ply`, `.vtk`, `.vtu`, etc.

## How It Works

- **Triangle Meshes**: Subdivides the mesh using the selected algorithm.
  - **Upsample**: Increases mesh resolution by splitting each triangle.
  - **Barycentric**: Performs barycentric subdivision.
  - **Loop**: Applies Loop subdivision for smooth surfaces.
- **Tetrahedral Meshes**: Extracts the boundary, subdivides the surface mesh, and re-tetrahedralizes the volume using `tetgen`.
