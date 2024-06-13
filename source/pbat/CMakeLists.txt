# Add sources
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    TYPE HEADERS
    FILES
    "Aliases.h"
    "DoctestLoadDLL.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "DoctestLoadDLL.cpp"
)

if(PBAT_BUILD_TESTS)
    # Regardless of static or shared build, PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    # will host the implementation of doctest.
    list(APPEND _pbat_doctest_macros "DOCTEST_CONFIG_IMPLEMENT")
    
    if(BUILD_SHARED_LIBS)
        list(APPEND _pbat_doctest_macros "DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL")
    endif()

    set_source_files_properties("DoctestLoadDLL.cpp"
        TARGET_DIRECTORY PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
        PROPERTIES
        COMPILE_DEFINITIONS ${_pbat_doctest_macros})
endif()

add_subdirectory(common)
add_subdirectory(fem)
add_subdirectory(math)
add_subdirectory(physics)
add_subdirectory(profiling)