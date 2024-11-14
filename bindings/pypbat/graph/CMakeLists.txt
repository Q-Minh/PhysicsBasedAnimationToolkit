target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Graph.h"
    "Partitioning.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Graph.cpp"
    "Partitioning.cpp"
)
