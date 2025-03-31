include(FetchContent)

find_package(doctest CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(range-v3 CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)

if(NOT TARGET Eigen3::Eigen)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen
        GIT_TAG 7fd305ecae2410714cde018cb6851f49138568c8
        GIT_PROGRESS TRUE
        SYSTEM
    )
    FetchContent_MakeAvailable(eigen)
endif()

if(PBAT_BUILD_PYTHON_BINDINGS AND NOT TARGET pybind11::headers)
    find_package(
        Python 
        COMPONENTS Development.Module 
        REQUIRED
    )
    FetchContent_Declare(
        _pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
        GIT_PROGRESS TRUE
        SYSTEM
    )
    FetchContent_MakeAvailable(_pybind11)
endif()

if(PBAT_ENABLE_PROFILER AND NOT TARGET Tracy::TracyClient)
    set(TRACY_ON_DEMAND ${PBAT_PROFILE_ON_DEMAND} CACHE BOOL "" FORCE)
    FetchContent_Declare(
        tracy
        GIT_REPOSITORY https://github.com/wolfpld/tracy.git
        GIT_TAG v0.10
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        SYSTEM
    )
    FetchContent_MakeAvailable(tracy)
endif()

if(NOT TARGET cpp-sort::cpp-sort)
    FetchContent_Declare(
        _cppsort
        # GIT_REPOSITORY https://github.com/Morwenn/cpp-sort.git
        # GIT_TAG 1.16.0
        GIT_REPOSITORY https://github.com/Q-Minh/cpp-sort
        GIT_TAG 7c87a8775ca37a80dd2a518a5f5da9b049555b60
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        SYSTEM
    )
    FetchContent_MakeAvailable(_cppsort)
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
    message(VERBOSE "PBAT -- Found MKL DLLs: ${PBAT_INTERNAL_MKL_DLLS}")
endif()

if(PBAT_USE_SUITESPARSE)
    # find_package(metis CONFIG)
    # if (${metis_FOUND})
    # get_target_property(_metis_configurations metis IMPORTED_CONFIGURATIONS)
    # foreach(_metis_configuration IN ITEMS ${_metis_configurations})
    # get_target_property(_metis_location metis IMPORTED_LOCATION_${_metis_configuration})
    # message(VERBOSE "Found metis: ${_metis_location}")
    # endforeach()
    # endif()
    find_package(suitesparse CONFIG REQUIRED)
endif()

if(PBAT_USE_METIS)
    find_package(metis CONFIG REQUIRED)
endif()

if(PBAT_USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(DEFINED CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        find_package(cuda-api-wrappers CONFIG REQUIRED)
        set_target_properties(cuda-api-wrappers::runtime-and-driver PROPERTIES SYSTEM ON)
    else()
        message(FATAL_ERROR "PBAT -- Could not find CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}")
    endif()
endif()

if(PBAT_BUILD_DOC)
    find_package(
        Doxygen 
        REQUIRED 
        OPTIONAL_COMPONENTS dot mscgen dia
    )
endif()