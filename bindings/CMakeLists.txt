Python_add_library(PhysicsBasedAnimationToolkit_Python MODULE WITH_SOABI)

# Convert space-separated PBAT_CMAKE_CXX_FLAGS to list (i.e. semi-colon separated string)
string(REPLACE " " ";" _pbat_cxx_flags ${PBAT_CMAKE_CXX_FLAGS})
target_compile_options(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${_pbat_cxx_flags}>
)

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

# Build the core parts of nanobind once
nanobind_build_library(nanobind SHARED)
set_target_properties(nanobind PROPERTIES SYSTEM ON)
# .. and link it against the nanobind parts
target_link_libraries(PhysicsBasedAnimationToolkit_Python PRIVATE 
    PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    nanobind
    Python::Module
)
# .. enable size optimizations
nanobind_opt_size(PhysicsBasedAnimationToolkit_Python)
# .. enable link time optimization
nanobind_lto(PhysicsBasedAnimationToolkit_Python)
# .. set the default symbol visibility to 'hidden'
nanobind_set_visibility(PhysicsBasedAnimationToolkit_Python)
# .. strip unneeded symbols and debug info from the binary (only active in release builds)
nanobind_strip(PhysicsBasedAnimationToolkit_Python)
# .. disable the stack protector
nanobind_disable_stack_protector(PhysicsBasedAnimationToolkit_Python)
# .. set the Python extension suffix
nanobind_extension(PhysicsBasedAnimationToolkit_Python)
# .. set important compilation flags
nanobind_compile_options(PhysicsBasedAnimationToolkit_Python)
# .. set important linker flags
nanobind_link_options(PhysicsBasedAnimationToolkit_Python)
# Statically link against libstdc++/libgcc when targeting musllinux
nanobind_musl_static_libcpp(PhysicsBasedAnimationToolkit_Python)

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
    PATTERN "*.exp" EXCLUDE
    PATTERN "*.lib" EXCLUDE
)

# Bundle external (i.e. not built by this project) shared library dependencies.
get_target_property(_pbat_link_libraries PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit LINK_LIBRARIES)

foreach(_pbat_link_library IN ITEMS ${_pbat_link_libraries})
    if(TARGET ${_pbat_link_library})
        get_target_property(_is_imported ${_pbat_link_library} IMPORTED)
        get_target_property(_type ${_pbat_link_library} TYPE)
        message(DEBUG "PBAT[python] -- LINK_LIBRARY=${_pbat_link_library} has TYPE=${_type}")

        if(${_is_imported} AND(_type STREQUAL "SHARED_LIBRARY"))
            message(VERBOSE "PBAT[python] -- SHARED_LIBRARY=${_pbat_link_library} is IMPORTED, bundling imported artifacts")
            install(IMPORTED_RUNTIME_ARTIFACTS ${_pbat_link_library} RUNTIME DESTINATION "pbatoolkit")
        endif()
    endif()
endforeach()

# Bundling MKL and CUDA dependencies is pretty expensive,
# so we should expect users to have installed those locally
# and have the DLLs in system search paths. Normally, the
# installers for MKL and CUDA should do that automatically.

# if(PBAT_USE_INTEL_MKL)
# message(VERBOSE "PBAT[python] -- Generating install rule for MKL DLLs ${PBAT_INTERNAL_MKL_DLLS}")
# install(FILES ${PBAT_INTERNAL_MKL_DLLS} DESTINATION "pbatoolkit")
# endif()

# if(PBAT_USE_CUDA)
# message(VERBOSE "PBAT[python] -- Generating install rule for CUDA DLLs located at ${CUDAToolkit_BIN_DIR}")
# install(
# DIRECTORY "${CUDAToolkit_BIN_DIR}/"
# DESTINATION "pbatoolkit"
# FILES_MATCHING
# PATTERN "*.dll"
# PATTERN "*.so")
# endif()
