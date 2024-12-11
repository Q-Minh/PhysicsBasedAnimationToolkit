target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "AxisAlignedBoundingBox.h"
    "Geometry.h"
    "MeshBoundary.h"
    "TetrahedralAabbHierarchy.h"
    "TriangleAabbHierarchy.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "AxisAlignedBoundingBox.cpp"
    "Geometry.cpp"
    "MeshBoundary.cpp"
    "TetrahedralAabbHierarchy.cpp"
    "TriangleAabbHierarchy.cpp"
)