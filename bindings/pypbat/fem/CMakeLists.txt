# add_library(PBAT_PyFem)
# set_target_properties(PBAT_PyFem
# PROPERTIES
# FOLDER "PhysicsBasedAnimationToolkit/bindings/fem"
# WINDOWS_EXPORT_ALL_SYMBOLS ON
# )

# list(APPEND _pbat_fem_types
# "Gradient"
# "HyperElasticPotential"
# "Jacobian"
# "LaplacianMatrix"
# "LoadVector"
# "MassMatrix"
# "Mesh"
# "ShapeFunctions"
# )

# set(_pbat_python_autogen_dir "gen")

# foreach(_pbat_fem_type IN ITEMS ${_pbat_fem_types})
# set(_pbat_python_fem_target "PBAT_PyFem_${_pbat_fem_type}")
# add_library(${_pbat_python_fem_target})
# set_target_properties(${_pbat_python_fem_target}
# PROPERTIES
# FOLDER "PhysicsBasedAnimationToolkit/bindings/fem/${_pbat_fem_type}"
# WINDOWS_EXPORT_ALL_SYMBOLS ON
# )
# target_link_libraries(${_pbat_python_fem_target}
# PUBLIC
# pybind11::headers
# Python::Module
# PRIVATE
# PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
# )

# execute_process(
# COMMAND ${Python_EXECUTABLE} bindings.py --cmake=1 --type=${_pbat_fem_type}
# WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
# OUTPUT_VARIABLE _pbat_fem_bindings_impl
# ERROR_VARIABLE _pbat_fem_bindings_error
# RESULT_VARIABLE _pbat_bindings_exit_code
# ECHO_ERROR_VARIABLE
# )

# if(NOT _pbat_bindings_exit_code EQUAL 0)
# message(FATAL_ERROR "Failed to generate FEM binding implementations.")
# endif()

# target_sources(${_pbat_python_fem_target}
# PUBLIC
# FILE_SET api
# TYPE HEADERS
# BASE_DIRS ${CMAKE_CURRENT_LIST_DIR}
# )

# target_sources(${_pbat_python_fem_target}
# PUBLIC
# FILE_SET api
# FILES
# "${_pbat_python_autogen_dir}/${_pbat_fem_type}.h"
# )

# target_sources(${_pbat_python_fem_target}
# PRIVATE
# "${_pbat_python_autogen_dir}/${_pbat_fem_type}.cpp"
# ${_pbat_fem_bindings_impl}
# )

# target_link_libraries(PBAT_PyFem
# PRIVATE
# ${_pbat_python_fem_target}
# )
# endforeach()

# target_link_libraries(PBAT_PyFem
# PUBLIC
# pybind11::headers
# Python::Module
# PRIVATE
# PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
# )
# target_sources(PBAT_PyFem
# PUBLIC
# FILE_SET api
# TYPE HEADERS
# BASE_DIRS ${CMAKE_CURRENT_LIST_DIR}
# )

# target_sources(PBAT_PyFem
# PUBLIC
# FILE_SET api
# FILES
# "Fem.h"
# )

# target_sources(PBAT_PyFem
# PRIVATE
# "Fem.cpp"
# )

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