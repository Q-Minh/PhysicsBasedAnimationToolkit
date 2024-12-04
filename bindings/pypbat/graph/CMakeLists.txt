target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Graph.h"
    "Partition.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Graph.cpp"
    "Partition.cpp"
)
