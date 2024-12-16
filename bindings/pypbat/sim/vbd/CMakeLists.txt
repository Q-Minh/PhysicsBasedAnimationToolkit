target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Data.h"
    "Integrator.h"
    "Vbd.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Data.cpp"
    "Integrator.cpp"
    "Vbd.cpp"
)

add_subdirectory(lod)
add_subdirectory(multigrid)