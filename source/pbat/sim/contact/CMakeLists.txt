target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "Constraints.h"
    "Contact.h"
    "MultibodyMeshMixedCcdDcd.h"
    "MultibodyTetrahedralMeshSystem.h"
    "MeshVertexTetrahedronDcd.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Constraints.cpp"
    "MultibodyMeshMixedCcdDcd.cpp"
    "MeshVertexTetrahedronDcd.cpp"
    "MultibodyTetrahedralMeshSystem.cpp"
)