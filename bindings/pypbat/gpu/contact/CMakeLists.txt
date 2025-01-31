target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Contact.h"
    "VertexTriangleMixedCcdDcd.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Contact.cpp"
    "VertexTriangleMixedCcdDcd.cpp"
)