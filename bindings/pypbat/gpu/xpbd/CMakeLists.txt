target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Integrator.h"
    "Xpbd.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Integrator.cpp"
    "Xpbd.cpp"
)