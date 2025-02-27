target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC 
    FILE_SET api
    FILES
    "Basis.h"
    "Concepts.h"
    "Roots.h"
    "Polynomial.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Basis.cpp"
    "Roots.cpp"
)