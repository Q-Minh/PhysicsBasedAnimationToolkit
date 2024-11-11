target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "Data.h"
    "Enums.h"
    "Integrator.h"
    "Kernels.h"
    "Xpbd.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Data.cpp"
    "Integrator.cpp"
    "Kernels.cpp"
)