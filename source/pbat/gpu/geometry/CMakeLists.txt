# target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
#     PUBLIC
#     FILE_SET api
#     FILES
# )
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    FILE_SET implementation
    FILES
    "SweepAndTiniestQueue.cuh"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "SweepAndTiniestQueue.cu"
)