target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Cholmod.h"
    "LinAlg.h"
    "Pardiso.h"
    "SimplicialLDLT.h"
    "SparsityPattern.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Cholmod.cpp"
    "LinAlg.cpp"
    "Pardiso.cpp"
    "SimplicialLDLT.cpp"
    "SparsityPattern.cpp"
)