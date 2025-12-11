target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "Constraints.h"
    "Contact.h"
    "Environment.h"
    "Friction.h"
    "MeshDynamics.h"
    "MeshSdfContact.h"
    "MultibodyMeshMixedCcdDcd.h"
    "MultiMesh.h"
    "Potentials.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Constraints.cpp"
    "Environment.cpp"
    "Friction.cpp"
    "MeshDynamics.cpp"
    "MeshSdfContact.cpp"
    "MultibodyMeshMixedCcdDcd.cpp"
    "MultiMesh.cpp"
    "Potentials.cpp"
)

add_subdirectory(ogc)