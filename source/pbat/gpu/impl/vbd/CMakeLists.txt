target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    FILE_SET implementation
    FILES
    "Integrator.cuh"
    "Kernels.cuh"
    "Vbd.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Integrator.cu"
    "Kernels.cu"
)
