target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "AxisAlignedBoundingBox.h"
    "Geometry.h"
    "TetrahedralAabbHierarchy.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "AxisAlignedBoundingBox.cpp"
    "Geometry.cpp"
    "TetrahedralAabbHierarchy.cpp"
)