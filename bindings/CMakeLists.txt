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

# Because our build tree puts all binaries in CMAKE_RUNTIME_OUTPUT_DIRECTORY_<CONFIG>, 
# we can install our Python bindings by directly copying the binaries.
install(
    DIRECTORY "$<TARGET_FILE_DIR:PhysicsBasedAnimationToolkit_Python>/" 
    DESTINATION "pbatoolkit"
    # On windows, only .dll is required at runtime, i.e. .lib and .exp files are not needed.
    PATTERN "*.exp" EXCLUDE
    PATTERN "*.lib" EXCLUDE
)

if(PBAT_USE_INTEL_MKL)
    message(VERBOSE "PBAT[python] -- Generating install rule for MKL DLLs ${PBAT_INTERNAL_MKL_DLLS}")
    install(FILES ${PBAT_INTERNAL_MKL_DLLS} DESTINATION "pbatoolkit")
endif()
