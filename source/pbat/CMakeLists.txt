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

if(BUILD_SHARED_LIBS AND PBAT_BUILD_TESTS)
    #[[
        These compile definitions need to be declared only once.
        See
        https://github.com/doctest/doctest/blob/master/examples/executable_dll_and_plugin/implementation_2.cpp
    ]]
    set_source_files_properties("DoctestLoadDLL.cpp"
        TARGET_DIRECTORY PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
        PROPERTIES
        COMPILE_DEFINITIONS "DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL;DOCTEST_CONFIG_IMPLEMENT")
endif()

add_subdirectory(common)
add_subdirectory(fem)
add_subdirectory(math)
add_subdirectory(physics)
add_subdirectory(profiling)