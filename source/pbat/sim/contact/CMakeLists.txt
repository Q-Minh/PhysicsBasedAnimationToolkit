target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "Contact.h"
    "MultibodyMeshMixedCcdDcd.h"
    "MultibodyTetrahedralMeshSystem.h"
    "MeshVertexTetrahedronDcd.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "MultibodyMeshMixedCcdDcd.cpp"
    "MeshVertexTetrahedronDcd.cpp"
    "MultibodyTetrahedralMeshSystem.cpp"
)