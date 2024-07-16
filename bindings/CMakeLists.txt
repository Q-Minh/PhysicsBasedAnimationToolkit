Python_add_library(PhysicsBasedAnimationToolkit_Python MODULE WITH_SOABI)

set_target_properties(PhysicsBasedAnimationToolkit_Python
    PROPERTIES
    OUTPUT_NAME "_pbat"
    EXPORT_NAME "Python"
    FOLDER "PhysicsBasedAnimationToolkit/bindings"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    pybind11::headers
)

if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(PhysicsBasedAnimationToolkit_Python)
endif()

add_subdirectory(pypbat)

# Make source tree appear as on file system in IDEs
get_target_property(_pbat_python_sources PhysicsBasedAnimationToolkit_Python SOURCES)
get_target_property(_pbat_python_headers PhysicsBasedAnimationToolkit_Python HEADER_SET_api)
source_group(
    TREE ${CMAKE_CURRENT_SOURCE_DIR}
    PREFIX "source"
    FILES ${_pbat_python_sources}
)
source_group(
    TREE ${CMAKE_CURRENT_SOURCE_DIR}
    PREFIX "headers"
    FILES ${_pbat_python_headers}
)

# Make sure Python module can find shared library dependencies
# in install tree on non-Windows platforms. For simplicity, we
# install shared dependencies in the same path as our Python
# bindings module, although we could install them to "bin".
# Some paths are explicitly ignored (see pyproject.toml).
# include(GNUInstallDirs)
list(APPEND _pbat_python_rpaths
    "$ORIGIN"
)
set_target_properties(PhysicsBasedAnimationToolkit_Python
    PROPERTIES
    INSTALL_RPATH ${_pbat_python_rpaths}
)

# Install rules for scikit-build-core.
install(
    TARGETS PhysicsBasedAnimationToolkit_Python
    LIBRARY DESTINATION "pbatoolkit"
)

# Install shared dependencies
function(install_transitive_dependencies target)
    get_target_property(_link_libraries ${target} LINK_LIBRARIES)
    message(VERBOSE "PBAT[python] -- ${target} LINK_LIBRARIES = ${_link_libraries}")

    foreach(_dependency IN ITEMS ${_link_libraries})
        if(TARGET ${_dependency})
            get_target_property(_library_type ${_dependency} TYPE)

            if(_library_type STREQUAL "SHARED_LIBRARY")
                message(VERBOSE "PBAT[python] -- Generating install rule for shared library ${_dependency}")
                install(FILES $<TARGET_FILE:${_dependency}> DESTINATION "pbatoolkit")
            endif()
        endif()
    endforeach()
endfunction()

install_transitive_dependencies(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit)
install_transitive_dependencies(PhysicsBasedAnimationToolkit_Python)

if(PBAT_USE_INTEL_MKL)
    message(VERBOSE "PBAT[python] -- Generating install rule for MKL DLLs ${PBAT_INTERNAL_MKL_DLLS}")
    install(FILES ${PBAT_INTERNAL_MKL_DLLS} DESTINATION "pbatoolkit")
endif()
