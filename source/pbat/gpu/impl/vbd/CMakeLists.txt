target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    FILE_SET implementation
    FILES
    "ChebyshevIntegrator.cuh"
    "Integrator.cuh"
    "Kernels.cuh"
    "TrustRegionIntegrator.cuh"
    "Vbd.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "ChebyshevIntegrator.cu"
    "Integrator.cu"
    "Kernels.cu"
    "TrustRegionIntegrator.cu"
)
