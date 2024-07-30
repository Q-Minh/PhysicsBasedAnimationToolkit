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
    "SweepAndTiniestQueue.cuh"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Primitives.cu"
    "SweepAndTiniestQueue.cu"
)