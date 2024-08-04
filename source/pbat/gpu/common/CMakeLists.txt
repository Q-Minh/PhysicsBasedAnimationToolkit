target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    FILE_SET implementation
    FILES
    "Buffer.cuh"
    "Queue.cuh"
    "Var.cuh"
)

target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Buffer.cu"
    "Queue.cu"
    "Var.cu"
)