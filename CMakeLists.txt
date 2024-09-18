cmake_minimum_required(VERSION 3.26)
project(PhysicsBasedAnimationToolkit VERSION 0.0.7 LANGUAGES CXX)
enable_testing()

# Configuration options
option(PBAT_BUILD_PYTHON_BINDINGS "Include Python binding module" OFF)
option(PBAT_BUILD_TESTS "Build tests" ${PROJECT_IS_TOP_LEVEL})
option(PBAT_ENABLE_PROFILER
    "Enable Tracy profiler client. Note that Microsoft Visual Studio users 
    need to disable the 'Edit And Continue' feature."
    OFF)
option(PBAT_PROFILE_ON_DEMAND
    "Build Tracy with TRACY_ON_DEMAND option"
    OFF)
option(PBAT_USE_INTEL_MKL "Link to Intel MKL for dense and sparse linear algebra." OFF)
option(PBAT_USE_SUITESPARSE
    "Link to SuiteSparse for sparse matrix decompositions. Overrides Intel MKL."
    OFF)
option(PBAT_BUILD_SHARED_LIBS "Build as shared library" OFF)
option(PBAT_USE_CUDA "Link to CUDA Toolkit for GPU API" OFF)

# Global settings
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Platform specific settings
include(platform)

# Build as static or shared
if(PBAT_BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
endif()

# No matter the platform, we decide the structure of our build tree. This should
# have no side-effect on library users.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Dependencies
# Note that this call must come after we have set BUILD_SHARED_LIBS
# in order to propagate the variable to our dependencies (i.e. Tracy).
include(dependencies)

# Sources
add_subdirectory(source)

if(PBAT_BUILD_PYTHON_BINDINGS)
    add_subdirectory(bindings)
endif()

if(PBAT_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Install
include(GNUInstallDirs)
set_target_properties(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_LIBDIR}
)
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
