target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC 
    FILE_SET api
    FILES
    "BranchAndBound.h"
    "LineSearch.h"
    "Newton.h"
    "Optimization.h"
    "TriangleConstrainedTrustRegionSr1.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "BranchAndBound.cpp"
    "LineSearch.cpp"
    "Newton.cpp"
    "TriangleConstrainedTrustRegionSr1.cpp"
)