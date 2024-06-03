add_library(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit)
add_library(PhysicsBasedAnimationToolkit::PhysicsBasedAnimationToolkit ALIAS PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit)
set_target_properties(
    PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PROPERTIES
    OUTPUT_NAME "PhysicsBasedAnimationToolkit"
    EXPORT_NAME "PhysicsBasedAnimationToolkit"
    FOLDER "PhysicsBasedAnimationToolkit"
    DEBUG_POSTFIX "_d"
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
    PBAT_ROOT="${PROJECT_SOURCE_DIR}"
)

# Define header file set
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_BINARY_DIR}
)

# Define sources+targets for our library and link to every dependency
add_subdirectory(pbat)
target_link_libraries(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    doctest::doctest
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    range-v3::concepts
    range-v3::meta
    range-v3::range-v3
    TBB::tbb
    $<TARGET_NAME_IF_EXISTS:MKL::MKL>
    $<TARGET_NAME_IF_EXISTS:SuiteSparse::suitesparseconfig>
    $<TARGET_NAME_IF_EXISTS:SuiteSparse::amd>
    $<TARGET_NAME_IF_EXISTS:SuiteSparse::cholmod>
    $<TARGET_NAME_IF_EXISTS:SuiteSparse::colamd>
    $<TARGET_NAME_IF_EXISTS:SuiteSparse::ccolamd>
    $<TARGET_NAME_IF_EXISTS:SuiteSparse::camd>
    $<TARGET_NAME_IF_EXISTS:SuiteSparse::ldl>
    $<TARGET_NAME_IF_EXISTS:SuiteSparse::umfpack>
    $<TARGET_NAME_IF_EXISTS:SuiteSparse::spqr>
    $<TARGET_NAME_IF_EXISTS:Tracy::TracyClient>
)

# Make source tree appear as on file system in IDEs
get_target_property(_pbat_sources PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit SOURCES)
get_target_property(_pbat_headers PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit HEADER_SET_api)
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
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    TYPE HEADERS
    FILES "${CMAKE_CURRENT_BINARY_DIR}/PhysicsBasedAnimationToolkitExport.h"
)