add_executable(PhysicsBasedAnimationToolkit_Benchmarks)
target_sources(PhysicsBasedAnimationToolkit_Benchmarks
    PRIVATE
    "benchmarks.cpp"
    "Sort.cpp"
)
target_link_libraries(PhysicsBasedAnimationToolkit_Benchmarks
    PRIVATE
    PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
)

if(PBAT_BUILD_TESTS)
    if(BUILD_SHARED_LIBS)
        set_source_files_properties("benchmarks.cpp"
            PROPERTIES
            COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL)
    endif()
else()
    set_source_files_properties("benchmarks.cpp"
        PROPERTIES
        COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENT)
endif()

target_link_libraries(PhysicsBasedAnimationToolkit_Benchmarks
    PRIVATE
    nanobench::nanobench
)

set_target_properties(PhysicsBasedAnimationToolkit_Benchmarks
    PROPERTIES
    FOLDER "PhysicsBasedAnimationToolkit/benchmarks"
    OUTPUT_NAME "PhysicsBasedAnimationToolkitBenchmarks"
    EXPORT_NAME "Benchmarks"
)

include(${PBAT_DOCTEST_MODULES_DIR}/doctest.cmake)
doctest_discover_tests(PhysicsBasedAnimationToolkit_Benchmarks)