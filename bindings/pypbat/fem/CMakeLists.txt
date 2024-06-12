target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "DivergenceVector.h"
    "fem.h"
    "Gradient.h"
    "HyperElasticPotential.h"
    "Jacobian.h"
    "LaplacianMatrix.h"
    "LoadVector.h"
    "MassMatrix.h"
    "Mesh.h"
    "ShapeFunctions.h"
    "For.h"
)

execute_process(
    COMMAND ${Python_EXECUTABLE} bindings.py --cmake=1
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
    OUTPUT_VARIABLE _pbat_fem_bindings_impl
    RESULT_VARIABLE _pbat_bindings_exit_code
)
if (NOT _pbat_bindings_exit_code EQUAL 0)
    message(FATAL_ERROR "Failed to generate FEM binding implementations.")
endif()

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "DivergenceVector.cpp"
    "fem.cpp"
    "Gradient.cpp"
    "HyperElasticPotential.cpp"
    "Jacobian.cpp"
    "LaplacianMatrix.cpp"
    "LoadVector.cpp"
    "MassMatrix.cpp"
    "Mesh.cpp"
    "ShapeFunctions.cpp"

    ${_pbat_fem_bindings_impl}
)