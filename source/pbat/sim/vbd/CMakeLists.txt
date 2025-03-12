target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "ChebyshevIntegrator.h"
    "Vbd.h"
    "Data.h"
    "Enums.h"
    "Kernels.h"
    "Integrator.h"
    "Mesh.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "ChebyshevIntegrator.cpp"
    "Data.cpp"
    "Kernels.cpp"
    "Integrator.cpp"
    "Mesh.cpp"
)

add_subdirectory(multigrid)