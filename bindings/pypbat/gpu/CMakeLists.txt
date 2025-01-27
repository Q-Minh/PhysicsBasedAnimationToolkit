target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Gpu.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Gpu.cpp"
)

add_subdirectory(common)
add_subdirectory(geometry)
add_subdirectory(xpbd)
add_subdirectory(vbd)
