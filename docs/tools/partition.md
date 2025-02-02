# Tetrahedral Mesh Partitioning Viewer

This script provides a visualization of tetrahedral mesh partitioning. It partitions a mesh into user-defined subdomains and displays the results using **Polyscope**, a visualization tool for 3D data.

---

## Features

- Partition a tetrahedral mesh into a specified number of parts.
- Visualize the mesh and partitioning results using **Polyscope**.
- Leverage **PBAToolkit** for graph-based mesh partitioning.

---

## Prerequisites

Ensure the following Python libraries are installed:

- `meshio`
- `numpy`
- `pbatoolkit`
- `polyscope`
- `argparse`

Install dependencies using `pip`:

```bash
pip install meshio numpy polyscope argparse
```

For **PBAToolkit**, follow the toolkit’s official installation instructions.

---

## Command-Line Arguments

| Argument         | Description                                     | Default |
|------------------|-------------------------------------------------|---------|
| `-i`, `--input` | Path to the input tetrahedral mesh file         | Required |
| `-n`, `--nparts`| Number of partitions to create                  | `8`     |

### Example Usage

```bash
python mesh_partitioning_viewer.py -i input_mesh.vtu -n 16
```

---

## Workflow

### 1. Load Mesh Data
The script reads the input mesh file using **meshio**:

```python
imesh = meshio.read(args.input)
V, C = imesh.points, imesh.cells_dict["tetra"]
```

- **V**: Vertex coordinates
- **C**: Tetrahedral cell indices

### 2. Generate Dual Graph
The **PBAToolkit** generates a dual graph representation of the mesh:

```python
GGT = pbat.graph.mesh_dual_graph(C, V.shape[0])
GGT.data[GGT.data < 3] = 0
GGT.eliminate_zeros()
```

This graph captures the connectivity of the mesh for partitioning.

### 3. Partition the Mesh
The `pbat.graph.partition` function partitions the mesh based on the dual graph:

```python
partitions = pbat.graph.partition(
    GGT.indptr, GGT.indices, weights, args.nparts,
    objective=pbat.graph.PartitioningObjective.MinEdgeCut,
    coarsening=pbat.graph.PartitioningCoarseningStrategy.SortedHeavyEdgeMatching,
    initializer=pbat.graph.InitialPartitioningStrategy.GreedyBisectionGrowing,
    refinement=pbat.graph.PartitioningRefinementStrategy.TwoSidedNodeFiducciaMattheyses,
    n_partition_trials=10,
    n_separators=10,
    n_refinement_iters=100,
    seed=0,
    minimize_degree=True,
    with_two_hop=True,
    contiguous_parts=True,
    identify_conn_comp=False
)
```

- **Objective**: Minimize edge cuts during partitioning.
- **Coarsening Strategy**: Combine graph edges with a heavy edge matching strategy.
- **Initializer**: Use a greedy bisection growing algorithm.
- **Refinement**: Optimize partitions using a two-sided Fiduccia-Mattheyses algorithm.

### 4. Visualize Results
**Polyscope** visualizes the mesh and its partitions:

```python
ps.set_program_name("Mesh partitioning")
ps.set_ground_plane_mode("shadow_only")
ps.set_SSAA_factor(4)
ps.set_up_dir("z_up")
ps.init()

vm = ps.register_volume_mesh("Mesh", V, C)
vm.add_scalar_quantity("Partitioning", partitions,
                       defined_on="cells", cmap="turbo", enabled=True)
ps.show()
```

- **Partitioning Visualization**: Assigns colors to partitions using the `turbo` colormap.
- **Mesh Display**: Uses Polyscope’s visualization capabilities.

---

## Notes

1. The script assumes the input file contains tetrahedral meshes.
2. Adjustments to partitioning strategies can be made by modifying the **PBAToolkit** arguments.
3. The default partitioning configuration minimizes edge cuts for balanced subdomains.

---

## References

- [PBAToolkit Documentation](https://github.com/path/to/pbatoolkit)
- [Polyscope Documentation](http://polyscope.run/)

