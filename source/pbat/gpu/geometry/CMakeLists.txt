# target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
# PUBLIC
# FILE_SET api
# FILES
# )
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    FILE_SET implementation
    FILES
    "AxisAlignedBoundingBoxes.cuh"
    "Primitives.cuh"
    "SweepAndTiniestQueue.cuh"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "AxisAlignedBoundingBoxes.cu"
    "Primitives.cu"
    "SweepAndTiniestQueue.cu"
)