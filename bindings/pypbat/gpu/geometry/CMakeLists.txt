target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Geometry.h"
    "Points.h"
    "Simplices.h"
    "SweepAndPrune.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Geometry.cpp"
    "Points.cpp"
    "Simplices.cpp"
    "SweepAndPrune.cpp"
)