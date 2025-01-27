target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Aabb.h"
    "Bodies.h"
    "Bvh.h"
    "BvhQuery.h"
    "Geometry.h"
    "Points.h"
    "Simplices.h"
    "SweepAndPrune.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Aabb.cpp"
    "Bodies.cpp"
    "Bvh.cpp"
    "BvhQuery.cpp"
    "Geometry.cpp"
    "Points.cpp"
    "Simplices.cpp"
    "SweepAndPrune.cpp"
)