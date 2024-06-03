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

add_subdirectory(common)
add_subdirectory(fem)
add_subdirectory(math)
add_subdirectory(physics)
add_subdirectory(profiling)