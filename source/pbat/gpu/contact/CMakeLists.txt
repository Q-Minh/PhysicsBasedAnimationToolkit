target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "Contact.h"
    "VertexTriangleMixedCcdDcd.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "VertexTriangleMixedCcdDcd.cu"
)
