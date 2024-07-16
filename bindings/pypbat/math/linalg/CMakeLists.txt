target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Cholmod.h"
    "LinAlg.h"
    "SimplicialLDLT.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Cholmod.cpp"
    "LinAlg.cpp"
    "SimplicialLDLT.cpp"
)