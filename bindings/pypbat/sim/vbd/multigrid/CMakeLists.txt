target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Hierarchy.h"
    "Integrator.h"
    "Level.h"
    "Multigrid.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Hierarchy.cpp"
    "Integrator.cpp"
    "Level.cpp"
    "Multigrid.cpp"
)
