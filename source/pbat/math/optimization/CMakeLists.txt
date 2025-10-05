target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC 
    FILE_SET api
    FILES
    "LineSearch.h"
    "Newton.h"
    "Optimization.h"
    "BranchAndBound.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "LineSearch.cpp"
    "Newton.cpp"
    "BranchAndBound.cpp"
)