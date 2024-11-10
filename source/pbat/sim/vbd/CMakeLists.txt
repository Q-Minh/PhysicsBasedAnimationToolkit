target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "Vbd.h"
    "Data.h"
    "Enums.h"
    "Kernels.h"
    "Integrator.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Data.cpp"
    "Kernels.cpp"
    "Integrator.cpp"
)