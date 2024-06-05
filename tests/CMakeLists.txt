add_executable(PhysicsBasedAnimationToolkit_Tests "tests.cpp")

if(BUILD_SHARED_LIBS)
    target_link_libraries(PhysicsBasedAnimationToolkit_Tests
        PRIVATE
        PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit)
    set_source_files_properties("tests.cpp"
        PROPERTIES
        COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL)
else()
    # Use WHOLE_ARCHIVE so that doctest tests implemented in PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    # are not optimized away.
    target_link_libraries(PhysicsBasedAnimationToolkit_Tests
        PRIVATE
        $<LINK_LIBRARY:WHOLE_ARCHIVE,PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit>)
    set_source_files_properties("tests.cpp"
        PROPERTIES
        COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENT)
endif()

set_target_properties(PhysicsBasedAnimationToolkit_Tests
    PROPERTIES
    FOLDER "PhysicsBasedAnimationToolkit/tests"
    OUTPUT_NAME "PhysicsBasedAnimationToolkitTests"
    EXPORT_NAME "Tests"
)
add_custom_command(
    TARGET PhysicsBasedAnimationToolkit_Tests
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_RUNTIME_DLLS:PhysicsBasedAnimationToolkit_Tests>
    $<TARGET_FILE_DIR:PhysicsBasedAnimationToolkit_Tests>
    COMMAND_EXPAND_LISTS
)
include(doctest)
doctest_discover_tests(PhysicsBasedAnimationToolkit_Tests)