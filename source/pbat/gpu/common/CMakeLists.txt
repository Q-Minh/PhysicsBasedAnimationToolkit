target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    FILE_SET implementation
    FILES
    "Buffer.cuh"
    "Cuda.cuh"
    "SynchronizedList.cuh"
    "Var.cuh"
)

target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Buffer.cu"
    "Cuda.cu"
    "SynchronizedList.cu"
    "Var.cu"
)