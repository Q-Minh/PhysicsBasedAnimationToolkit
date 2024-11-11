target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Math.h"
    "MomentFitting.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Math.cpp"
    "MomentFitting.cpp"
)

add_subdirectory(linalg)