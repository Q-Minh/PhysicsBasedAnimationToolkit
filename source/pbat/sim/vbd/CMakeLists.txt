target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "AcceleratedAndersonIntegrator.h"
    "AndersonIntegrator.h"
    "ChebyshevIntegrator.h"
    "Data.h"
    "Enums.h"
    "Integrator.h"
    "Kernels.h"
    "Mesh.h"
    "NesterovIntegrator.h"
    "Vbd.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "AcceleratedAndersonIntegrator.cpp"
    "AndersonIntegrator.cpp"
    "ChebyshevIntegrator.cpp"
    "Data.cpp"
    "Integrator.cpp"
    "Kernels.cpp"
    "Mesh.cpp"
    "NesterovIntegrator.cpp"
)

add_subdirectory(multigrid)