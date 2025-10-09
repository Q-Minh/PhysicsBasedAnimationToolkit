include(FetchContent)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

find_package(fmt CONFIG REQUIRED)
find_package(range-v3 CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)

if(NOT TARGET doctest::doctest)
    FetchContent_Declare(
        _doctest
        GIT_REPOSITORY https://github.com/doctest/doctest.git
        GIT_TAG 3a01ec37828affe4c9650004edb5b304fb9d5b75
        GIT_PROGRESS TRUE
        SYSTEM
    )
    FetchContent_MakeAvailable(_doctest)
    get_target_property(PBAT_DOCTEST_SOURCE_DIR doctest::doctest SOURCE_DIR)
    message(VERBOSE "PBAT -- Doctest source directory: ${PBAT_DOCTEST_SOURCE_DIR}")
    set(PBAT_DOCTEST_MODULES_DIR ${PBAT_DOCTEST_SOURCE_DIR}/scripts/cmake/)
endif()

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

find_package(HDF5 CONFIG REQUIRED)

if(NOT TARGET HighFive::HighFive)
    set(HIGHFIVE_FIND_HDF5 OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        HighFive
        GIT_REPOSITORY https://github.com/highfive-devs/highfive.git
        GIT_TAG e950d07369344a5821b8a63697de3bd84664186e
        GIT_PROGRESS TRUE
        SYSTEM
    )
    FetchContent_MakeAvailable(HighFive)
endif()

if(PBAT_BUILD_PYTHON_BINDINGS AND NOT TARGET nanobind::headers)
    find_package(
        Python
        COMPONENTS Interpreter Development.Module
        REQUIRED
    )

    # TODO: Change back to the official nanobind repository once PR for stubgen lib_path is merged.
    FetchContent_Declare(
        _nanobind
        GIT_REPOSITORY https://github.com/Doekin/nanobind.git
        GIT_TAG stubgen_win_dll
        GIT_PROGRESS TRUE
        SYSTEM
    )
    FetchContent_MakeAvailable(_nanobind)
endif()

if(PBAT_ENABLE_PROFILER AND NOT TARGET Tracy::TracyClient)
    set(TRACY_ON_DEMAND ${PBAT_PROFILE_ON_DEMAND} CACHE BOOL "" FORCE)

    # TODO: Change back to the official tracy repository once PR for nvcc compilation fix is merged.
    FetchContent_Declare(
        tracy
        GIT_REPOSITORY https://github.com/Q-Minh/tracy
        GIT_TAG master
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        SYSTEM
    )
    FetchContent_MakeAvailable(tracy)
endif()

if(NOT TARGET cpp-sort::cpp-sort)
    FetchContent_Declare(
        _cppsort
        GIT_REPOSITORY https://github.com/Morwenn/cpp-sort.git
        GIT_TAG 2.x.y-stable
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
    include(CheckLanguage)
    check_language(C)

    if(DEFINED CMAKE_C_COMPILER)
        enable_language(C)
    else()
        message(FATAL_ERROR "PBAT -- Could not find CMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
    endif()

    find_package(OpenMP REQUIRED COMPONENTS C)
    find_package(CHOLMOD CONFIG REQUIRED)
endif()

if(PBAT_USE_METIS)
    find_package(metis CONFIG REQUIRED)
endif()

if(PBAT_USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(DEFINED CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit 12.8.0...<13.0.0 REQUIRED)
        FetchContent_Declare(
            _caw
            GIT_REPOSITORY https://github.com/eyalroz/cuda-api-wrappers.git
            GIT_TAG v0.8.1
            GIT_SHALLOW TRUE
            GIT_PROGRESS TRUE
            SYSTEM
        )
        FetchContent_MakeAvailable(_caw)
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