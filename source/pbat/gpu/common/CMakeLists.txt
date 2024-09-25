target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    FILE_SET implementation
    FILES
    "Buffer.cuh"
    "Cuda.cuh"
    "Morton.cuh"
    "Queue.cuh"
    "Stack.cuh"
    "SynchronizedList.cuh"
    "Var.cuh"
)

target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Buffer.cu"
    "Cuda.cu"
    "Morton.cu"
    "Queue.cu"
    "Stack.cu"
    "SynchronizedList.cu"
    "Var.cu"
)