add_library(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit)
add_library(PhysicsBasedAnimationToolkit::PhysicsBasedAnimationToolkit ALIAS PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit)
set_target_properties(
    PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PROPERTIES
    OUTPUT_NAME "PhysicsBasedAnimationToolkit"
    EXPORT_NAME "PhysicsBasedAnimationToolkit"
    FOLDER "PhysicsBasedAnimationToolkit"
    DEBUG_POSTFIX "_d"
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
)

# Most of our implementation code is templated, so we expose transitive dependencies (i.e. PUBLIC)
target_compile_features(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit PUBLIC cxx_std_20)
target_compile_definitions(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    $<$<BOOL:${PBAT_BUILD_TESTS}>:PBAT_HAS_DOCTEST>
    $<$<NOT:$<BOOL:${PBAT_BUILD_TESTS}>>:DOCTEST_CONFIG_DISABLE>
    $<$<BOOL:${PBAT_ENABLE_PROFILER}>:PBAT_HAS_TRACY_PROFILER>
    $<$<BOOL:${PBAT_USE_INTEL_MKL}>:EIGEN_USE_MKL_ALL>
    $<$<BOOL:${PBAT_USE_INTEL_MKL}>:PBAT_USE_INTEL_MKL>
    $<$<BOOL:${PBAT_USE_SUITESPARSE}>:PBAT_USE_SUITESPARSE>
    $<$<BOOL:${PBAT_USE_CUDA}>:PBAT_USE_CUDA>
    $<$<BOOL:${PBAT_USE_CUDA}>:EIGEN_NO_CUDA>
    PBAT_ROOT="${PROJECT_SOURCE_DIR}"
)

# Define header file sets
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_BINARY_DIR}
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    FILE_SET implementation
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_LIST_DIR}
)

# Define sources+targets for our library and link to every dependency
add_subdirectory(pbat)
target_link_libraries(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    doctest::doctest
    Eigen3::Eigen
    fmt::fmt
    range-v3::concepts
    range-v3::meta
    range-v3::range-v3
    TBB::tbb
)

if(PBAT_ENABLE_PROFILER)
    target_link_libraries(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
        PUBLIC
        Tracy::TracyClient
    )
endif()

if(PBAT_USE_INTEL_MKL)
    target_link_libraries(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
        PUBLIC
        MKL::MKL
    )
endif()

if(PBAT_USE_SUITESPARSE)
    target_link_libraries(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
        PUBLIC
        SuiteSparse::suitesparseconfig
        SuiteSparse::amd
        SuiteSparse::cholmod
        SuiteSparse::colamd
        SuiteSparse::ccolamd
        SuiteSparse::camd
        SuiteSparse::ldl
        SuiteSparse::umfpack
        SuiteSparse::spqr
    )
endif()

if(PBAT_USE_CUDA)
    target_link_libraries(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
        PRIVATE
        CUDA::toolkit
        CUDA::cudart
        cuda-api-wrappers::runtime-and-driver
    )
    set_target_properties(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
        PROPERTIES
        CUDA_ARCHITECTURES native
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    get_target_property(_pbat_sources PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit SOURCES)

    foreach(_source IN ITEMS ${_pbat_sources})
        get_filename_component(_extension ${_source} LAST_EXT)

        if(_extension STREQUAL ".cu")
            set_source_files_properties(${_source}
                PROPERTIES
                COMPILE_OPTIONS "--expt-relaxed-constexpr;--extended-lambda"
                COMPILE_DEFINITIONS "_SILENCE_ALL_CXX20_DEPRECATION_WARNINGS")
        endif()
    endforeach()
endif()

# Make source tree appear as on file system in IDEs
get_target_property(_pbat_sources PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit SOURCES)
get_target_property(_pbat_headers PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit HEADER_SET_api)
get_target_property(_pbat_private_headers PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit HEADER_SET_implementation)
list(APPEND _pbat_headers ${_pbat_private_headers})
source_group(
    TREE ${CMAKE_CURRENT_SOURCE_DIR}
    PREFIX "source"
    FILES ${_pbat_sources}
)
source_group(
    TREE ${CMAKE_CURRENT_SOURCE_DIR}
    PREFIX "headers"
    FILES ${_pbat_headers}
)

# Generate shared library export macros
# Note that the following code must come after the previous call to source_group,
# otherwise the export file will not play well with source_group, since it resides
# in the ${CMAKE_CURRENT_BINARY_DIR} folder, not in the current source tree.
include(GenerateExportHeader)
generate_export_header(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    EXPORT_MACRO_NAME "PBAT_API"
    EXPORT_FILE_NAME "PhysicsBasedAnimationToolkitExport.h"
    DEPRECATED_MACRO_NAME "PBAT_DEPRECATED"
    STATIC_DEFINE "PBAT_STATIC_DEFINE"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    TYPE HEADERS
    FILES "${CMAKE_CURRENT_BINARY_DIR}/PhysicsBasedAnimationToolkitExport.h"
)