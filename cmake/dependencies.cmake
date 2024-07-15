find_package(doctest CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(range-v3 CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)

if(PBAT_BUILD_PYTHON_BINDINGS)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
endif()

include(FetchContent)

if(PBAT_ENABLE_PROFILER)
    set(TRACY_ON_DEMAND ${PBAT_PROFILE_ON_DEMAND})
    FetchContent_Declare(
        tracy
        GIT_REPOSITORY https://github.com/wolfpld/tracy.git
        GIT_TAG v0.10
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(tracy)
endif()

if(PBAT_USE_INTEL_MKL)
    find_package(MKL CONFIG REQUIRED)

    # Many of the MKL DLLs are not exposed as transitive dependencies of MKL::MKL
    # and MKL::mkl_core, but our library target PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    # needs them. Thus, we manually search for the missing DLLs (i.e. all of them) and
    # append them to some global variable PBAT_INTERNAL_MKL_DLLS. Targets which needs
    # those DLLs can link to those missing dependencies by using this variable.
    get_target_property(_mkl_imported_location MKL::mkl_core IMPORTED_LOCATION)
    cmake_path(GET _mkl_imported_location PARENT_PATH _mkl_shared_library_directory)
    cmake_path(APPEND _mkl_shared_library_directory "mkl_*.dll" OUTPUT_VARIABLE _mkl_dlls_glob)
    file(
        GLOB _mkl_shared_libraries
        LIST_DIRECTORIES OFF
        "${_mkl_dlls_glob}")
    set(PBAT_INTERNAL_MKL_DLLS ${_mkl_shared_libraries})
    message(VERBOSE "Found MKL DLLs: ${PBAT_INTERNAL_MKL_DLLS}")
endif()

if(PBAT_USE_SUITESPARSE)
    # find_package(metis CONFIG)
    # if (${metis_FOUND})
    # get_target_property(_metis_configurations metis IMPORTED_CONFIGURATIONS)
    # foreach(_metis_configuration IN ITEMS ${_metis_configurations})
    # get_target_property(_metis_location metis IMPORTED_LOCATION_${_metis_configuration})
    # message(STATUS "Found metis: ${_metis_location}")
    # endforeach()
    # endif()
    find_package(suitesparse CONFIG REQUIRED)
endif()
