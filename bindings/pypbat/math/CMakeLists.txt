target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Math.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Math.cpp"
)

add_subdirectory(linalg)