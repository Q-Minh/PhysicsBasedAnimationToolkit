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
endif()

# On Windows, make sure DLL dependencies (i.e. runtime artifacts)
# end up in runtime output directory. For some reason, many of the 
# MKL DLLs are not copied to the output directory, so we enforce it. 
# Only mkl_core.2.dll and mkl_intel_thread.2.dll are copied. 
if(WIN32 AND PBAT_USE_INTEL_MKL)
    if (PBAT_INTERNAL_MKL_DLLS)
        add_custom_command(
            TARGET PhysicsBasedAnimationToolkit_Tests
            COMMAND
            ${CMAKE_COMMAND} -E copy_if_different
            ${PBAT_INTERNAL_MKL_DLLS}
            $<TARGET_FILE_DIR:PhysicsBasedAnimationToolkit_Tests>
            COMMAND_EXPAND_LISTS
        )
    endif()
endif()

set_target_properties(PhysicsBasedAnimationToolkit_Tests
    PROPERTIES
    FOLDER "PhysicsBasedAnimationToolkit/tests"
    OUTPUT_NAME "PhysicsBasedAnimationToolkitTests"
    EXPORT_NAME "Tests"
)

include(${PBAT_DOCTEST_MODULES_DIR}/doctest.cmake)
doctest_discover_tests(PhysicsBasedAnimationToolkit_Tests)