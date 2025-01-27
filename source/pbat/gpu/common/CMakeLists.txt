target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "Buffer.h"
    "Common.h"
)

target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Buffer.cu"
)
