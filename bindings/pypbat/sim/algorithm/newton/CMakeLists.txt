target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Newton.h"
)
target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Newton.cpp"
)
