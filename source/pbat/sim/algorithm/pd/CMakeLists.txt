target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "PD.h"
    "ValanisLandelQuasiNewton.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "PD.cpp"
    "ValanisLandelQuasiNewton.cpp"
)