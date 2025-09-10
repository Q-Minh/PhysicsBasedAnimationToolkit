target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "Config.h"
    "HessianProduct.h"
    "Integrator.h"
    "Newton.h"
    "Preconditioner.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Config.cpp"
    "HessianProduct.cpp"
    "Integrator.cpp"
    "Newton.cpp"
    "Preconditioner.cpp"
)