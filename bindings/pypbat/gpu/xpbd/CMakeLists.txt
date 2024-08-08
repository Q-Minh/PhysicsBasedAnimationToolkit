target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Xpbd.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Xpbd.cpp"
)