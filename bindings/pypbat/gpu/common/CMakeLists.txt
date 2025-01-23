target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Buffer.h"
    "Common.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Buffer.cpp"
    "Common.cpp"
)