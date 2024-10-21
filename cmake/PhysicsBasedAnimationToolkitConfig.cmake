include(CMakeFindDependencyMacro)

find_dependency(doctest)
find_dependency(Eigen3)
find_dependency(fmt)
find_dependency(range-v3)
find_dependency(TBB)
find_dependency(OpenMP)

unset(_pbat_find_pkg_args)

if(${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
    list(APPEND _pbat_find_pkg_args QUIET)
endif()

if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
    list(APPEND _pbat_find_pkg_args REQUIRED)
endif()

if(PBAT_BUILD_PYTHON_BINDINGS)
    find_package(Python COMPONENTS Interpreter Development.Module ${_pbat_find_pkg_args})
    find_package(pybind11 CONFIG ${_pbat_find_pkg_args})
endif()

if(PBAT_USE_INTEL_MKL)
    find_dependency(MKL)
endif()

if(PBAT_USE_SUITESPARSE)
    find_dependency(suitesparse)
endif()

if(PBAT_USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if (NOT DEFINED CMAKE_CUDA_COMPILER)
        if (${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
            message(FATAL_ERROR "PBAT -- Could not find CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}")
        endif()
    else()
        enable_language(CUDA)
        find_dependency(CUDAToolkit)
        find_dependency(cuda-api-wrappers)
    endif()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/PhysicsBasedAnimationToolkit_Targets.cmake)