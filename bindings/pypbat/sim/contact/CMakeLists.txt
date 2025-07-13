target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Contact.h"
    "MeshVertexTetrahedronDcd.h"
    "MultibodyMeshMixedCcdDcd.h"
)
target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Contact.cpp"
    "MultibodyMeshMixedCcdDcd.cpp"
    "MeshVertexTetrahedronDcd.cpp"
)
