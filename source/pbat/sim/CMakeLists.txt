target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "Sim.h"
)
add_subdirectory(vbd)
add_subdirectory(xpbd)