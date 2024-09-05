target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    FILE_SET implementation
    FILES
    "Buffer.cuh"
    "Morton.cuh"
    "Queue.cuh"
    "Stack.cuh"
    "Var.cuh"
)

target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Buffer.cu"
    "Morton.cu"
    "Queue.cu"
    "Stack.cu"
    "Var.cu"
)