target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "Bvh.h"
    "BvhQuery.h"
    "Primitives.h"
    "SweepAndPrune.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Bvh.cu"
    "BvhQuery.cu"
    "Primitives.cu"
    "SweepAndPrune.cu"
)

add_subdirectory(impl)