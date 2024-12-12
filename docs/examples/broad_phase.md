# 3D Broad Phase Collision Detection with PBAT

This repository contains a Python script for broad phase collision detection using the **Physics Based Animation Toolkit (PBAT)**. It includes two algorithms for collision detection between 3D meshes:
- **Sweep and Prune**
- **Bounding Volume Hierarchy (BVH)**

The GUI is built using **Polyscope**, a 3D visualization framework, allowing real-time interaction with the meshes.

## Features

- Supports tetrahedral meshes in `.vtk`, `.msh`, `.obj`, and other common formats.
- Animates and visualizes broad phase collision detection in 3D.
- Allows switching between collision detection algorithms.
- Includes a configurable user interface for controlling the animation, speed, and algorithm.
- Optional export of frame-by-frame screenshots.

## Requirements

Before running the script, ensure the following packages are installed:

- **pbatoolkit** (Python bindings of PBAT)
- **Polyscope** (for 3D visualization)
- **Meshio** (for reading and writing mesh files)
- **NumPy** (for numerical operations)
- **ImGui** (Polyscope's GUI framework)

Install dependencies via `pip`:

```bash
pip install numpy==1.26 scipy==1.14 meshio==5.3.5 libigl==v2.5.1 polyscope==2.2.1ilupp==1.0.2 ipctk==1.2.0 networkx==3.3
```

## Usage

### Command-line Arguments

The script accepts the following arguments:

- `-i` / `--input` (required): Path to the input tetrahedral mesh file (supported formats include `.vtk`, `.obj`, `.msh`, etc.)
- `-o` / `--output` (optional): Directory path where screenshots will be saved when exporting frames. Default is the current directory.
- `-t` / `--translation` (optional): Vertical translation applied to the second mesh in the animation. Default is `0.1`.

### Example Usage

To run the script with an input mesh:

```bash
python broad_phase.py -i path/to/mesh.vtk
```

To specify an output directory for saving screenshots:

```bash
python broad_phase.py -i path/to/mesh.vtk -o output/directory
```

## Interacting with the GUI

Once the script is running, the Polyscope window will open, showing the meshes and the results of collision detection. The GUI includes the following controls:

- **Algorithm**: Choose between `Sweep and Prune` or `Bounding Volume Hierarchy` for collision detection.
- **Box expansion**: Adjust the amount of expansion applied to the bounding boxes for collision detection.
- **Speed**: Adjust the speed of the mesh animation.
- **Export**: Toggle to save screenshots of the frames.
- **Animate**: Toggle to start or stop the animation.
- **Step**: Manually step through the animation frame by frame.

## Output

The script will update the visualization in real time, displaying the active overlapping simplices (tetrahedrons) in the meshes. The overlapping regions will be highlighted, and users can switch between algorithms to compare performance.

### Screenshot Exporting

When `Export` is enabled, the script will save screenshots of each frame to the specified output directory in `.png` format. The file names will be formatted as `{frame_number}.png`.