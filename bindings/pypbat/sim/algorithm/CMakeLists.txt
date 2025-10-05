target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Algorithm.h"
)
target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Algorithm.cpp"
)

add_subdirectory(newton)
add_subdirectory(pd)