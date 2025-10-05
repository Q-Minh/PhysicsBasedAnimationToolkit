target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Integration.h"
    "Bdf.h"
)
target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Integration.cpp"
    "Bdf.cpp"
)