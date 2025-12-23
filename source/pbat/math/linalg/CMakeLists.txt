add_subdirectory(mini)

target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "CholmodSupport.h"
    "EigenSolver.h"
    "LinAlg.h"
    "FilterEigenvalues.h"
    "SelectionMatrix.h"
    "SparsityPattern.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "CholmodSupport.cpp"
    "EigenSolver.cpp"
    "FilterEigenvalues.cpp"
    "SelectionMatrix.cpp"
    "SparsityPattern.cpp"
)