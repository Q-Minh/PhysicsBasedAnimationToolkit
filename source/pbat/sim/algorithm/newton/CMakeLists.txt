target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "Config.h"
    "Core.h"
    "Integrator.h"
    "Newton.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Config.cpp"
    "Core.cpp"
    "Integrator.cpp"
    "Newton.cpp"
)