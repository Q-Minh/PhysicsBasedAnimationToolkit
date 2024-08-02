target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "Primitives.h"
    "SweepAndPrune.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    FILE_SET implementation
    FILES
    "PrimitivesImpl.cuh"
    "SweepAndPruneImpl.cuh"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Primitives.cu"
    "PrimitivesImpl.cu"
    "SweepAndPrune.cu"
    "SweepAndPruneImpl.cu"
)