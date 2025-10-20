target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Anderson.h"
    "Chebyshev.h"
    "Core.h"
    "Broyden.h"
    "Vbd.h"
)
target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Anderson.cpp"
    "Chebyshev.cpp"
    "Core.cpp"
    "Broyden.cpp"
    "Vbd.cpp"
)
