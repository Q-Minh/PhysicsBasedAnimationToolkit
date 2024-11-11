target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Integrator.h"
    "Vbd.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Integrator.cpp"
    "Vbd.cpp"
)