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

# DEPRECATED: Multigrid code is old, not used and not maintained.
# add_subdirectory(multigrid)