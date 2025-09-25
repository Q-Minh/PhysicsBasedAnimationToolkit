target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Sim.h"
)
target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Sim.cpp"
)

add_subdirectory(algorithm)
add_subdirectory(contact)
add_subdirectory(dynamics)
add_subdirectory(integration)
add_subdirectory(vbd)
add_subdirectory(xpbd)
