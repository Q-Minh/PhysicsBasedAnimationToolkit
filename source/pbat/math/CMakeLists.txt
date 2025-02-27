target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "Concepts.h"
    "GaussQuadrature.h"
    "IntegerArithmeticChecks.h"
    "LinearOperator.h"
    "Math.h"
    "MomentFitting.h"
    "Rational.h"
    "SymmetricQuadratureRules.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "GaussQuadrature.cpp"
    "IntegerArithmeticChecks.cpp"
    "LinearOperator.cpp"
    "MomentFitting.cpp"
    "Rational.cpp"
    "SymmetricQuadratureRules.cpp"
)

add_subdirectory(linalg)
add_subdirectory(optimization)
add_subdirectory(polynomial)