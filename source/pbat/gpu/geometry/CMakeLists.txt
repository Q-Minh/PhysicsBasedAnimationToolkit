# target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
# PUBLIC
# FILE_SET api
# FILES
# )
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    FILE_SET implementation
    FILES
    "Primitives.cuh"
    "SweepAndPrune.h"
    "SweepAndPruneImpl.cuh"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Primitives.cu"
    "SweepAndPrune.cu"
    "SweepAndPruneImpl.cu"
)