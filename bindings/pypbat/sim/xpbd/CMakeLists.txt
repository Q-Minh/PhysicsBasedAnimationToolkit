target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Data.h"
    "Integrator.h"
    "Xpbd.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Data.cpp"
    "Integrator.cpp"
    "Xpbd.cpp"
)