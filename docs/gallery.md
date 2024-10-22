# Gallery

Below, we show a few examples of what can be done in just a few lines of code using [`pbatoolkit`](https://pypi.org/project/pbatoolkit/) and Python. Code can be found [here](../examples/index.md).

## Real-time Hyper Elasticity Dynamics

Our GPU implementation of the eXtended Position Based Dynamics (XPBD) algorithm simulates a ~324k element FEM elastic mesh interactively with contact.

<p float="left">
    <img src="_static/imgs/gpu.xpbd.bvh.gif" width="250" alt="A 162k element armadillo mesh is dropped on top of another duplicate, but fixed, armadillo mesh on the bottom." />
</p>

## Inter-penetration Free Elastodynamic Contact

Combining [`pbatoolkit`](https://pypi.org/project/pbatoolkit/)’s FEM+elasticity features and the [`IPC Toolkit`](https://ipctk.xyz/) results in guaranteed inter-penetration free contact dynamics between deformable bodies.

<p float="left">
    <img src="_static/imgs/ipc.bar.stacks.gif" width="250" alt="A stack of bending beams fall on top of each other, simulated via Incremental Potential Contact (IPC)." />
</p>

## Modal Analysis

The hyper elastic beam's representative deformation modes, i.e., its low frequency eigen vectors, are animated as time continuous signals.

<p float="left">
    <img src="_static/imgs/beam.modes.gif" width="250" alt="Unconstrained hyper elastic beam's eigen frequencies" />
</p>

## GPU Broad Phase Collision Detection

Real-time collision detection between 2 large scale meshes (~324k tetrahedra) is accelerated by highly parallel implementations of the [sweep and prune](https://en.wikipedia.org/wiki/Sweep_and_prune) algorithm, or [linear bounding volume hierarchies](https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf).

<p float="left">
    <img src="_static/imgs/gpu.broadphase.gif" width="250" alt="Broad phase collision detection on the GPU between 2 moving tetrahedral meshes" />
</p>

## Harmonic Interpolation

A smooth (harmonic) function is constructed on [Entei](https://bulbapedia.bulbagarden.net/wiki/Entei_(Pok%C3%A9mon)), required to evaluate to `1` on its paws, and `0` at the top of its tail, using piece-wise linear (left) and quadratic (right) shape functions. Its isolines are displayed as black curves.

<p float="left">
  <img src="_static/imgs/entei.harmonic.interpolation.order.1.png" width="250" alt="Harmonic interpolation on Entei model using linear shape functions" />
  <img src="_static/imgs/entei.harmonic.interpolation.order.2.png" width="250" alt="Harmonic interpolation on Entei model using quadratic shape functions" /> 
</p>

## Heat Method for Geodesic Distance Computation

Approximate geodesic distances are computed from the top center vertex of [Metagross](https://bulbapedia.bulbagarden.net/wiki/Metagross_(Pok%C3%A9mon)) by diffusing heat from it (left), and recovering a function whose gradient matches the normalized heat's negative gradient. Its isolines are displayed as black curves.

<p float="left">
  <img src="_static/imgs/metagross.heat.source.png" width="250" alt="Heat source on top center of metagross model" />
  <img src="_static/imgs/metagross.heat.geodesics.png" width="250" alt="Reconstructed single source geodesic distance" /> 
</p>

## Mesh Smoothing via Diffusion

Fine details of Godzilla's skin are smoothed out by diffusing `x,y,z` coordinates in time.

<p float="left">
    <img src="_static/imgs/godzilla.diffusion.smoothing.gif" width="250" alt="Godzilla model with fine details being smoothed out via diffusion" />
</p>

## Profiling Statistics

Computation details are gathered when using [`pbatoolkit`](https://pypi.org/project/pbatoolkit/) and consulted in the [Tracy](https://github.com/wolfpld/tracy) profiling server GUI.

<p float="left">
    <img src="_static/imgs/profiling.statistics.png" alt="Profiling statistics widget in Tracy server" />
</p>
