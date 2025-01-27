target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "Adjacency.h"
    "Color.h"
    "Enums.h"
    "Graph.h"
    "Mesh.h"
    "Partition.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Adjacency.cpp"
    "Color.cpp"
    "Mesh.cpp"
    "Partition.cpp"
)