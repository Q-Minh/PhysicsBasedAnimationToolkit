target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "PD.h"
    "ValanisLandelQuasiNewton.h"
)
target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "PD.cpp"
    "ValanisLandelQuasiNewton.cpp"
)
