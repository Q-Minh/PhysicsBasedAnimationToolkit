target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Contact.h"
    "MeshDynamics.h"
    "MeshSdfContact.h"
    "MultibodyMeshMixedCcdDcd.h"
    "MultiMesh.h"
)
target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Contact.cpp"
    "MeshDynamics.cpp"
    "MeshSdfContact.cpp"
    "MultibodyMeshMixedCcdDcd.cpp"
    "MultiMesh.cpp"
)

add_subdirectory(ogc)