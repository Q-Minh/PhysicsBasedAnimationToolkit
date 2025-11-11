target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Optimization.h"
    "LineSearch.h"
    "Newton.h"
    "TriangleConstrainedTrustRegionSr1.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Optimization.cpp"
    "LineSearch.cpp"
    "Newton.cpp"
    "TriangleConstrainedTrustRegionSr1.cpp"
)
