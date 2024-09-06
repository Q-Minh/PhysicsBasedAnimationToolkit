# target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
# PUBLIC
# FILE_SET api
# FILES
# )
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    FILE_SET implementation
    FILES
    "Xpbd.h"
    "XpbdImpl.cuh"
    "XpbdImplKernels.cuh"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Xpbd.cu"
    "XpbdImpl.cu"
    "XpbdImplKernels.cu"
)