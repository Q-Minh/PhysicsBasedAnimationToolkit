target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "InitializationStrategy.h"
    "Vbd.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    FILE_SET implementation
    FILES
    "VbdImpl.cuh"
    "VbdImplKernels.cuh"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Vbd.cu"
    "VbdImpl.cu"
    "VbdImplKernels.cu"
)

add_subdirectory(tests)