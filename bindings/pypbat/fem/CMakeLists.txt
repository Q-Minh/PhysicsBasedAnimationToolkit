add_library(PhysicsBasedAnimationToolkit_Python_Fem)
set_target_properties(PhysicsBasedAnimationToolkit_Python_Fem
    PROPERTIES
    FOLDER "PhysicsBasedAnimationToolkit/bindings/fem"
)

list(APPEND _pbat_fem_types
    "Gradient"
    "HyperElasticPotential"
    "Jacobian"
    "LaplacianMatrix"
    "LoadVector"
    "MassMatrix"
    "Mesh"
    "ShapeFunctions"
)

set(_pbat_python_autogen_dir "pbatautogen")

foreach(_pbat_fem_type IN ITEMS ${_pbat_fem_types})
    set(_pbat_python_fem_target "PhysicsBasedAnimationToolkit_Python_Fem_${_pbat_fem_type}")
    add_library(${_pbat_python_fem_target})
    set_target_properties(${_pbat_python_fem_target}
        PROPERTIES
        FOLDER "PhysicsBasedAnimationToolkit/bindings/fem/${_pbat_fem_type}"
    )
    target_link_libraries(${_pbat_python_fem_target}
        PRIVATE
        PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
        Python::Module
        pybind11::headers
    )

    execute_process(
        COMMAND ${Python_EXECUTABLE} bindings.py --cmake=1 --type=${_pbat_fem_type}
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
        OUTPUT_VARIABLE _pbat_fem_bindings_impl
        ERROR_VARIABLE _pbat_fem_bindings_error
        RESULT_VARIABLE _pbat_bindings_exit_code
    )

    if(NOT _pbat_bindings_exit_code EQUAL 0)
        message(FATAL_ERROR "Failed to generate FEM binding implementations.")
    endif()

    target_sources(${_pbat_python_fem_target}
        PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_LIST_DIR}
    )

    target_sources(${_pbat_python_fem_target}
        PUBLIC
        FILE_SET api
        FILES
        "${_pbat_python_autogen_dir}/${_pbat_fem_type}.h"
    )

    target_sources(${_pbat_python_fem_target}
        PRIVATE
        "${_pbat_python_autogen_dir}/${_pbat_fem_type}.cpp"
        ${_pbat_fem_bindings_impl}
    )

    target_link_libraries(PhysicsBasedAnimationToolkit_Python_Fem
        PRIVATE
        ${_pbat_python_fem_target}
        Python::Module
        pybind11::headers
    )
endforeach()

target_link_libraries(PhysicsBasedAnimationToolkit_Python_Fem
    PRIVATE
    PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    Python::Module
    pybind11::headers
)
target_sources(PhysicsBasedAnimationToolkit_Python_Fem
    PUBLIC
    FILE_SET api
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_LIST_DIR}
)

target_sources(PhysicsBasedAnimationToolkit_Python_Fem
    PUBLIC
    FILE_SET api
    FILES
    "fem.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python_Fem
    PRIVATE
    "fem.cpp"
)