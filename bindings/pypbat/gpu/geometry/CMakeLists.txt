target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Bvh.h"
    "Geometry.h"
    "Points.h"
    "Simplices.h"
    "SweepAndPrune.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Bvh.cpp"
    "Geometry.cpp"
    "Points.cpp"
    "Simplices.cpp"
    "SweepAndPrune.cpp"
)