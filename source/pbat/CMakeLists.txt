# Add sources
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    TYPE HEADERS
    FILES
    "Aliases.h"
    "DoctestLoadDLL.h"
    "Pbat.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "DoctestLoadDLL.cpp"
)

if(PBAT_BUILD_TESTS)
    # Regardless of static or shared build, PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    # will host the implementation of doctest.
    if(BUILD_SHARED_LIBS)
        set_source_files_properties("DoctestLoadDLL.cpp"
            TARGET_DIRECTORY PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
            PROPERTIES
            COMPILE_DEFINITIONS "DOCTEST_CONFIG_IMPLEMENT;DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL")
    else()
        set_source_files_properties("DoctestLoadDLL.cpp"
            TARGET_DIRECTORY PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
            PROPERTIES
            COMPILE_DEFINITIONS "DOCTEST_CONFIG_IMPLEMENT")
    endif()
endif()

add_subdirectory(common)
add_subdirectory(fem)
add_subdirectory(geometry)
add_subdirectory(math)
add_subdirectory(physics)
add_subdirectory(profiling)