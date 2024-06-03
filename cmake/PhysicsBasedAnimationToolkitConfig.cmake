include(CMakeFindDependencyMacro)

find_dependency(doctest)
find_dependency(Eigen3)
find_dependency(OpenMP)
find_dependency(range-v3)
find_dependency(TBB)

if(PBAT_BUILD_PYTHON_BINDINGS)
    if(${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
        list(APPEND _pbat_find_pkg_args QUIET)
    endif()

    if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
        list(APPEND _pbat_find_pkg_args REQUIRED)
    endif()

    find_package(Python COMPONENTS Interpreter Development ${_pbat_find_pkg_args})
    find_package(pybind11 CONFIG ${_pbat_find_pkg_args})
endif()

if(PBAT_USE_INTEL_MKL)
    find_package(MKL)
endif()

if(PBAT_USE_SUITESPARSE)
    find_package(suitesparse)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/PhysicsBasedAnimationToolkit_Targets.cmake)