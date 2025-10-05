target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Aabb.h"
    "Bvh.h"
    "Geometry.h"
    "SweepAndPrune.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Aabb.cpp"
    "Bvh.cpp"
    "Geometry.cpp"
    "SweepAndPrune.cpp"
)