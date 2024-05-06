cmake_minimum_required(VERSION 3.23)
project(pba-core VERSION 0.0.1 LANGUAGES CXX)

# Configuration options
option(PBA_CORE_BUILD_PYTHON_BINDINGS "Include Python binding module" OFF)
option(PBA_CORE_BUILD_TESTS "Build tests" OFF)
option(PBA_CORE_ENABLE_PROFILER 
    "Enable Tracy profiler client. Note that Microsoft Visual Studio users 
    need to disable the 'Edit And Continue' feature." 
    OFF)
option(PBA_CORE_USE_INTEL_MKL "Link to Intel MKL for dense and sparse linear algebra." OFF)
option(PBA_CORE_USE_SUITESPARSE 
    "Link to SuiteSparse for sparse matrix decompositions. Overrides Intel MKL." 
    OFF)

# Global settings
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Core library
set(pba_core "pba-core")

# If we want to support both static and shared library builds, we need to 
# handle them separately, since doctest behaves differently for static vs 
# shared. In the static case, we need to create an object library and make 
# the doctest executable depend on those objects so that our library tests 
# get registered. 
# See https://github.com/doctest/doctest/blob/master/doc/markdown/faq.md#why-are-my-tests-in-a-static-library-not-getting-registered.
add_library(${pba_core} SHARED)
target_include_directories(${pba_core} 
PUBLIC 
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        >
)

file(
    GLOB_RECURSE _pba_core_sources 
    LIST_DIRECTORIES OFF 
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" 
    CONFIGURE_DEPENDS 
    "src/*.cpp"
    )
file(
    GLOB_RECURSE _pba_core_headers 
    LIST_DIRECTORIES OFF 
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" 
    CONFIGURE_DEPENDS 
    "include/*.h")
target_sources(${pba_core} PRIVATE ${_pba_core_headers} ${_pba_core_sources})
source_group(
TREE 
    "${CMAKE_CURRENT_SOURCE_DIR}" 
FILES 
    ${_pba_core_headers} 
    ${_pba_core_sources}
)

set_target_properties(${pba_core} 
PROPERTIES 
    FOLDER ${pba_core}
    WINDOWS_EXPORT_ALL_SYMBOLS ON 
    DEBUG_POSTFIX "_d" 
    POSITION_INDEPENDENT_CODE ON
)
target_compile_features(${pba_core} PUBLIC cxx_std_20)
target_compile_definitions(${pba_core} 
PRIVATE 
    PBA_CORE_ROOT="${CMAKE_CURRENT_LIST_DIR}"
)

# Dependencies
find_package(doctest CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(EnTT CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(range-v3 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)

target_link_libraries(${pba_core} 
PUBLIC
    doctest::doctest 
    Eigen3::Eigen
    range-v3::concepts
    range-v3::meta
    range-v3::range-v3
PRIVATE
    EnTT::EnTT
    OpenMP::OpenMP_CXX
    spdlog::spdlog
    TBB::tbb
)

if (PBA_CORE_BUILD_PYTHON_BINDINGS)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)

    file(
        GLOB_RECURSE _pba_core_bindings_sources 
        LIST_DIRECTORIES OFF 
        RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" 
        CONFIGURE_DEPENDS 
        "bindings/src/*.cpp")
    file(
        GLOB_RECURSE _pba_core_bindings_headers 
        LIST_DIRECTORIES OFF 
        RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" 
        CONFIGURE_DEPENDS 
        "bindings/include/*.h")

    list(APPEND _pba_core_bindings_all 
        ${_pba_core_bindings_sources} 
        ${_pba_core_bindings_headers})

    set(pba_core_python_bindings "pypba")
    # NOTE: Will Tracy profiling be available while using pba-core 
    # via its Python bindings? Better hope so!
    pybind11_add_module(${pba_core_python_bindings} 
        MODULE 
        EXCLUDE_FROM_ALL 
        ${_pba_core_bindings_all})

    target_link_libraries(${pba_core_python_bindings} 
    PRIVATE 
        ${pba_core})
endif()

if (PBA_CORE_BUILD_TESTS)
    set(pba_core_tests "pba-core-tests")
    # target_compile_definitions(${pba_core} 
    # PRIVATE 
    #     DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL 
    #     DOCTEST_CONFIG_IMPLEMENT)
    add_executable(${pba_core_tests} ${CMAKE_CURRENT_LIST_DIR}/tests/tests.cpp)
    target_link_libraries(${pba_core_tests} 
    PRIVATE 
        ${pba_core})
    set_target_properties(${pba_core_tests} PROPERTIES FOLDER "tests")
    add_custom_command(TARGET ${pba_core_tests} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
            $<TARGET_RUNTIME_DLLS:${pba_core_tests}> 
            $<TARGET_FILE_DIR:${pba_core_tests}>
        COMMAND_EXPAND_LISTS
    )
else()
    target_compile_definitions(${pba_core} PRIVATE DOCTEST_CONFIG_DISABLE)
endif()

if (PBA_CORE_ENABLE_PROFILER)
    find_package(Threads REQUIRED)

    find_package(Tracy CONFIG REQUIRED)
    # NOTE: TracyClient dependency is made public so that it propagates to 
    # other linked libraries/executables, i.e. the Python bindings. 
    # WARNING: However, the vcpkg package for Tracy only supports static linking.
    # Therefore, if some other DLL links against pba-core, and the 
    # Tracy::TracyClient dependency is a PUBLIC static library, will 
    # the Tracy::TracyClient be duplicated in each DLL? If so, then this 
    # will be a problem for the Tracy profiler.
    target_link_libraries(${pba_core} 
    PUBLIC
        Tracy::TracyClient 
    )
    if (UNIX)
        target_link_libraries(${pba_core} 
        PUBLIC
            Threads::Threads 
            ${CMAKE_DL_LIBS}
        )
    endif()
    # Making TRACY_ENABLE public so that we can profile the core library 
    # when it is used by some downstream code.
    target_compile_definitions(${pba_core} 
    PUBLIC 
        TRACY_ENABLE
        TRACY_IMPORTS
        PBA_CORE_HAS_TRACY_PROFILER)
endif()

if (PBA_CORE_USE_INTEL_MKL)
    find_package(MKL CONFIG REQUIRED)
    target_link_libraries(${pba_core} PRIVATE MKL::MKL)
    target_compile_definitions(${pba_core} 
    PRIVATE 
        EIGEN_USE_MKL_ALL
        PBA_CORE_USE_INTEL_MKL)
endif()

if (PBA_CORE_USE_SUITESPARSE)
    find_package(suitesparse CONFIG REQUIRED)
    target_link_libraries(${pba_core} 
    PRIVATE 
        SuiteSparse::suitesparseconfig
        SuiteSparse::amd
        SuiteSparse::cholmod
        SuiteSparse::colamd
        SuiteSparse::ccolamd
        SuiteSparse::camd
        SuiteSparse::ldl
        SuiteSparse::umfpack
        SuiteSparse::spqr)
    target_compile_definitions(${pba_core} PRIVATE PBA_CORE_USE_SUITESPARSE)
endif()
