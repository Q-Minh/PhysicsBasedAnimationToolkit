target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    FILE_SET implementation
    FILES
    "Contact.h"
    "VertexTriangleMixedCcdDcd.cuh"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "VertexTriangleMixedCcdDcd.cu"
)