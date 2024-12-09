target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Data.h"
    "Hierarchy.h"
    "Integrator.h"
    "Level.h"
    "MultiScaleIntegrator.h"
    "Prolongation.h"
    "Restriction.h"
    "Smoother.h"
    "Vbd.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Data.cpp"
    "Hierarchy.cpp"
    "Integrator.cpp"
    "Level.cpp"
    "MultiScaleIntegrator.cpp"
    "Prolongation.cpp"
    "Restriction.cpp"
    "Smoother.cpp"
    "Vbd.cpp"
)

add_subdirectory(multigrid)