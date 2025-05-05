target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Dynamics.h"
    "FemElastoDynamics.h"
)
target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Dynamics.cpp"
    "FemElastoDynamics.cpp"
)