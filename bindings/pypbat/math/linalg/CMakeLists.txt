target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "LinAlg.h"
    "SimplicialLDLT.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "LinAlg.cpp"
    "SimplicialLDLT.cpp"
)