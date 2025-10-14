target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Cholmod.h"
    "Enums.h"
    "LinAlg.h"
    "Pardiso.h"
    "FilterEigenvalues.h"
    "SimplicialLDLT.h"
    "SparsityPattern.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Cholmod.cpp"
    "Enums.cpp"
    "LinAlg.cpp"
    "Pardiso.cpp"
    "FilterEigenvalues.cpp"
    "SimplicialLDLT.cpp"
    "SparsityPattern.cpp"
)