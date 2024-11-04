target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "Xpbd.h"
    "Data.h"
    "Kernels.h"
    "Integrator.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Data.cpp"
    "Kernels.cpp"
    "Integrator.cpp"
)