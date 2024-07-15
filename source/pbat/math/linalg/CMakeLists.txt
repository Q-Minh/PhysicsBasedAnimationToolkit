target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "Cholmod.h"
    "LinAlg.h"
    "SparsityPattern.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Cholmod.cpp"
    "SparsityPattern.cpp"
)