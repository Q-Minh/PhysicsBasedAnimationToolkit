target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Sdf.h"
    "BinaryNode.h"
    "Composite.h"
    "Primitive.h"
    "Transform.h"
    "UnaryNode.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Sdf.cpp"
    "BinaryNode.cpp"
    "Composite.cpp"
    "Primitive.cpp"
    "Transform.cpp"
    "UnaryNode.cpp"
)