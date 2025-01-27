target_sources(
    PUBLIC PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    FILE_SET api
    FILES
    "Optimization.h"
    "BranchAndBound.h"
)
target_sources(
    PRIVATE
    "BranchAndBound.cpp"
)