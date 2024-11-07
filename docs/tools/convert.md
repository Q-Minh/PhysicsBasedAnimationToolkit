# Mesh Format Converter

A simple command-line tool to convert mesh files between different formats using `meshio`.

## Description

This script reads a mesh file in one format and writes it out in another, facilitating easy conversion between various mesh formats supported by `meshio`.

## Usage

Run the script from the command line with the required input and output file paths:

```bash
python mesh_converter.py -i input_mesh.ext -o output_mesh.ext
```

### Arguments

- `-i`, `--input`: Path to the input mesh file (required).
- `-o`, `--output`: Path to the output mesh file (required).

## Requirements

- Python 3.x
- `meshio` library

## Installation

Install the necessary Python package:

```bash
pip install meshio
```

## Supported Formats

The tool supports various mesh formats, including:

- **Input Formats**: `.obj`, `.off`, `.stl`, `.ply`, `.vtk`, `.vtu`, `.msh`, etc.
- **Output Formats**: Same as input formats.

For a full list of supported formats, refer to the [meshio documentation](https://github.com/nschloe/meshio).