target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "DivergenceVector.h"
    "fem.h"
    "HyperElasticPotential.h"
    "Jacobian.h"
    "LaplacianMatrix.h"
    "LoadVector.h"
    "MassMatrix.h"
    "Mesh.h"
    "ShapeFunctions.h"
    "For.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "DivergenceVector.cpp"
    "fem.cpp"
    "HyperElasticPotential.cpp"
    "Jacobian.cpp"
    "LaplacianMatrix.cpp"
    "LoadVector.cpp"
    "MassMatrix.cpp"
    "Mesh.cpp"
    "ShapeFunctions.cpp"
)