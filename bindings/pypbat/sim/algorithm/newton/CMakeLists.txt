target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Newton.h"
    "Core.h"
)
target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Newton.cpp"
    "Core.cpp"
)
