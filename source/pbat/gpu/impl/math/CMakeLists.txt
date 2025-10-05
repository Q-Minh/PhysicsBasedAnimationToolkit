target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    FILE_SET implementation
    FILES
    "Blas.cuh"
    "LinearSolver.cuh"
    "Math.h"
    "Matrix.cuh"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Blas.cu"
    "LinearSolver.cu"
    "Matrix.cu"
)
