target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Optimization.h"
    "LineSearch.h"
    "Newton.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Optimization.cpp"
    "LineSearch.cpp"
    "Newton.cpp"
)
