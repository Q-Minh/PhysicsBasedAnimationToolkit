target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Fem.h"
    "Gradient.h"
    "HyperElasticPotential.h"
    "Jacobian.h"
    "Laplacian.h"
    "MassMatrix.h"
    "Mesh.h"
    "ShapeFunctions.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Fem.cpp"
    "Gradient.cpp"
    "Jacobian.cpp"
    "HyperElasticPotential.cpp"
    "Laplacian.cpp"
    "MassMatrix.cpp"
    "Mesh.cpp"
    "ShapeFunctions.cpp"
)