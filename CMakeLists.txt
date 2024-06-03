cmake_minimum_required(VERSION 3.26)
project(PhysicsBasedAnimationToolkit VERSION 0.0.1 LANGUAGES CXX)
enable_testing()

# Configuration options
option(PBAT_BUILD_PYTHON_BINDINGS "Include Python binding module" OFF)
option(PBAT_BUILD_TESTS "Build tests" ${PROJECT_IS_TOP_LEVEL})
option(PBAT_ENABLE_PROFILER
    "Enable Tracy profiler client. Note that Microsoft Visual Studio users 
    need to disable the 'Edit And Continue' feature."
    OFF)
option(PBAT_USE_INTEL_MKL "Link to Intel MKL for dense and sparse linear algebra." OFF)
option(PBAT_USE_SUITESPARSE
    "Link to SuiteSparse for sparse matrix decompositions. Overrides Intel MKL."
    OFF)
option(PBAT_BUILD_SHARED_LIBS "Build as shared library" OFF)

# Global settings
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
list(APPEND CMAKE_MODULE_PATH "cmake")

# Build as static or shared
if(PBAT_BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
endif()

# Dependencies
include(dependencies)

add_subdirectory(source)

if(PBAT_BUILD_PYTHON_BINDINGS)
    add_subdirectory(bindings)
endif()

if(PBAT_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(PBAT_USE_INTEL_MKL)
    # NOTE:
    # I don't know the cleanest way to setup DLL paths on Windows that handles DLL dependencies
    # and their own transitive DLL dependencies. The commented code beneath handles a specific
    # case of the mkl_def.2.dll file not being copied into the test executable's folder when
    # using the Intel MKL vcpkg port.
    # For some reason, the vcpkg intel MKL package does not expose the mkl_def shared library
    # executable, which is needed for linking. We write a patch to find the mkl_def executable
    # in the same directory as mkl_core and then copy it to our binary folder on build.
    # get_target_property(_mkl_imported_location MKL::mkl_core IMPORTED_LOCATION)
    # message(STATUS "MKL IMPORTED LOCATION=${_mkl_imported_location}")
    # cmake_path(
    # GET _mkl_imported_location
    # PARENT_PATH _mkl_shared_library_directory)
    # message(STATUS "MKL SHARED LIBRARY DIRECTORY=${_mkl_shared_library_directory}")
    # file(
    # GLOB _external_shared_libraries
    # LIST_DIRECTORIES FALSE
    # CONFIGURE_DEPENDS

    # # Are there other possible names for shared libraries on other platforms?
    # "${_mkl_shared_library_directory}/mkl_def.2.dll"
    # "${_mkl_shared_library_directory}/mkl_def.2.so"
    # "${_mkl_shared_library_directory}/libmkl_def.2.so")
    # message(STATUS "MKL DEF LIBRARIES=${_external_shared_libraries}")
    # list(GET _external_shared_libraries 0 _mkl_def_shared_library)
    # message(STATUS "MKL DEF SHARED LIBRARY=${_mkl_def_shared_library}")
    # cmake_path(
    # APPEND _mkl_shared_library_directory _mkl_def_shared_library
    # OUTPUT_VARIABLE _path_to_mkl_def)

    # if(PBAT_BUILD_TESTS)
    # add_custom_command(TARGET PhysicsBasedAnimationToolkit_Tests POST_BUILD
    # COMMAND ${CMAKE_COMMAND} -E copy_if_different
    # ${_mkl_def_shared_library}
    # $<TARGET_FILE_DIR:PhysicsBasedAnimationToolkit_Tests>
    # COMMAND_EXPAND_LISTS
    # )
    # endif()

    # if(PBAT_BUILD_PYTHON_BINDINGS)
    # add_custom_command(TARGET PhysicsBasedAnimationToolkit_Python POST_BUILD
    # COMMAND ${CMAKE_COMMAND} -E copy_if_different
    # ${_mkl_def_shared_library}
    # $<TARGET_FILE_DIR:PhysicsBasedAnimationToolkit_Python>
    # COMMAND_EXPAND_LISTS
    # )
    # endif()
endif()

# Install
include(GNUInstallDirs)
install(
    TARGETS
    PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    EXPORT PhysicsBasedAnimationToolkit_Targets
    FILE_SET api
)
install(
    EXPORT PhysicsBasedAnimationToolkit_Targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PhysicsBasedAnimationToolkit
    NAMESPACE PhysicsBasedAnimationToolkit::
    FILE PhysicsBasedAnimationToolkit_Targets.cmake
)

include(CMakePackageConfigHelpers)
install(
    FILES cmake/PhysicsBasedAnimationToolkitConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PhysicsBasedAnimationToolkit
)
write_basic_package_version_file(
    PhysicsBasedAnimationToolkitConfigVersion.cmake
    COMPATIBILITY ExactVersion
)
