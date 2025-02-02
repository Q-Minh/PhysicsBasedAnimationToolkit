# GPU Module

This module provides Python bindings for GPU-accelerated computations and submodules related to physical simulations, including geometry, VBD (Virtual Bond Dynamics), and XPBD (Extended Position-Based Dynamics). These submodules leverage GPU power to perform high-performance simulations.

## Submodules

### **geometry**
Provides GPU-accelerated geometric computations, including bounding boxes and bounding volume hierarchies (BVH).

- **Functions**:
  - `aabb`: Computes the axis-aligned bounding box (AABB) of input points.
  - `bvh`: Computes the bounding volume hierarchy (BVH) for a mesh using cell types like Triangle or Tetrahedron.

### **vbd**
This submodule offers GPU-optimized methods for Virtual Bond Dynamics (VBD), a technique used for simulating soft body dynamics and material deformation.

- **Functions**:  

### **xpbd**
The XPBD submodule implements the Extended Position-Based Dynamics method for simulating physical systems, particularly in scenarios involving deformable objects and constraints.

- **Functions**:  