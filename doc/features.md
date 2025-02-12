\page features Features

- [Finite Element Method](https://hplgit.github.io/fem-book/doc/pub/book/pdf/fem-book-4print-2up.pdf#page=47.07) (FEM) meshes and operators
  - Dimensions `1,2,3`
  - [Lagrange shape functions](https://doc.comsol.com/5.3/doc/com.comsol.help.comsol/comsol_api_xmesh.40.4.html) of order `1,2,3`
  - Line, triangle, quadrilateral, tetrahedron and hexahedron elements
- [Hyperelastic material models](https://en.wikipedia.org/wiki/Hyperelastic_material)
  - Saint-Venant Kirchhoff
  - [Stable Neo-Hookean](https://graphics.pixar.com/library/StableElasticity/paper.pdf)
- Polynomial [quadrature rules](https://en.wikipedia.org/wiki/Numerical_integration)
  - [Simplices in dimensions `1,2,3`](https://lsec.cc.ac.cn/~tcui/myinfo/paper/quad.pdf)
  - [Gauss-Legendre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature)
- Spatial query acceleration data structures
  - [Bounding volume hierarchy](https://en.wikipedia.org/wiki/Bounding_volume_hierarchy) for triangles (2D+3D) and tetrahedra (3D)
    - Nearest neighbours
    - Overlapping primitive pairs
    - Point containment
- GPU algorithms
  - [Vertex Block Descent](https://graphics.cs.utah.edu/research/projects/vbd/vbd-siggraph2024.pdf) (VBD)
  - [eXtended Position Based Dynamics](https://mmacklin.com/xpbd.pdf) (XPBD)
  - Broad phase collision detection
    - [Sweep and Prune](https://en.wikipedia.org/wiki/Sweep_and_prune)
    - [Linear Bounding Volume Hierarchy](https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf)
  - Fixed-size linear algebra library for kernel programming
- Seamless profiling integration via [Tracy](https://github.com/wolfpld/tracy)

<div class="section_buttons">

| Previous       |            Next |
|:---------------|----------------:|
| \ref index     | \ref quickstart |

</div>