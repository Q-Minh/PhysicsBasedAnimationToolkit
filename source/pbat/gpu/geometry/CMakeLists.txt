target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "Aabb.h"
    "Bvh.h"
    "Geometry.h"
    "Primitives.h"
    "SweepAndPrune.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Aabb.cu"
    "Bvh.cu"
    "Primitives.cu"
    "SweepAndPrune.cu"
)
