target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "Bvh.h"
    "Primitives.h"
    "SweepAndPrune.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    FILE_SET implementation
    FILES
    "BvhImpl.cuh"
    "PrimitivesImpl.cuh"
    "SweepAndPruneImpl.cuh"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Bvh.cu"
    "BvhImpl.cu"
    "Primitives.cu"
    "PrimitivesImpl.cu"
    "SweepAndPrune.cu"
    "SweepAndPruneImpl.cu"
)