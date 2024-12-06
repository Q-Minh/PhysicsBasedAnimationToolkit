target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Adjacency.h"
    "Color.h"
    "Graph.h"
    "Mesh.h"
    "Partition.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Adjacency.cpp"
    "Color.cpp"
    "Graph.cpp"
    "Mesh.cpp"
    "Partition.cpp"
)
