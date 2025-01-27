target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC 
    FILE_SET api
    FILES
    "Optimization.h"
    "BranchAndBound.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "BranchAndBound.cpp"
)