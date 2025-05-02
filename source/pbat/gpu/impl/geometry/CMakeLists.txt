target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    FILE_SET implementation
    FILES
    "Aabb.cuh"
    "Bvh.cuh"
    "Geometry.h"
    "Morton.cuh"
    "SweepAndPrune.cuh"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Aabb.cu"
    "Bvh.cu"
    "Morton.cu"
    "SweepAndPrune.cu"
)
