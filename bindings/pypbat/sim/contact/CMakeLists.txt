target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Contact.h"
    "MultibodyMeshMixedCcdDcd.h"
    "MultibodyTetrahedralMeshSystem.h"
)
target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Contact.cpp"
    "MultibodyMeshMixedCcdDcd.cpp"
    "MultibodyTetrahedralMeshSystem.cpp"
)
