target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "Core.h"
    "PD.h"
    "ValanisLandelQuasiNewton.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Core.cpp"
    "PD.cpp"
    "ValanisLandelQuasiNewton.cpp"
)