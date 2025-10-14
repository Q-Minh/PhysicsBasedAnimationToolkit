target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Profiling.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Profiling.cpp"
)