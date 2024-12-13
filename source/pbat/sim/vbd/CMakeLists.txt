target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "Vbd.h"
    "Data.h"
    "Enums.h"
    "Kernels.h"
    "Integrator.h"
    "Mesh.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Data.cpp"
    "Kernels.cpp"
    "Integrator.cpp"
    "Mesh.cpp"
)

add_subdirectory(lod)
add_subdirectory(multigrid)