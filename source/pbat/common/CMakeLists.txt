target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES 
        "Common.h"
        "Concepts.h"
        "ConstexprFor.h"
        "Eigen.h"
        "Hash.h"
        "Indexing.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
PRIVATE
    "Eigen.cpp"
    "Indexing.cpp"
)