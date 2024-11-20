target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Data.h"
    "Integrator.h"
    "Restriction.h"
    "Vbd.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Data.cpp"
    "Integrator.cpp"
    "Restriction.cpp"
    "Vbd.cpp"
)