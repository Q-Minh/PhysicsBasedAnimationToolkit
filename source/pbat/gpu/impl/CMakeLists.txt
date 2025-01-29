target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    FILE_SET implementation
    FILES
    "Impl.h"
)

add_subdirectory(common)
add_subdirectory(contact)
add_subdirectory(geometry)
add_subdirectory(vbd)
add_subdirectory(xpbd)