target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Contact.h"
    "MeshSdfContact.h"
    "MultibodyMeshMixedCcdDcd.h"
    "MultiMesh.h"
    "OffsetGeometryContact.h"
)
target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Contact.cpp"
    "MeshSdfContact.cpp"
    "MultibodyMeshMixedCcdDcd.cpp"
    "MultiMesh.cpp"
    "OffsetGeometryContact.cpp"
)
