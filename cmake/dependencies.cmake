find_package(doctest CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(range-v3 CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)

if(PBAT_BUILD_PYTHON_BINDINGS)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
endif()

include(FetchContent)
if(PBAT_ENABLE_PROFILER)
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
endif()

if(PBAT_USE_SUITESPARSE)
    find_package(suitesparse CONFIG REQUIRED)
endif()