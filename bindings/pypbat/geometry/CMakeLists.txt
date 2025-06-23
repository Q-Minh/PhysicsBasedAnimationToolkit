target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "AxisAlignedBoundingBox.h"
    "HashGrid.h"
    "Geometry.h"
    "MeshBoundary.h"
    "TetrahedralAabbHierarchy.h"
    "TriangleAabbHierarchy.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "AxisAlignedBoundingBox.cpp"
    "HashGrid.cpp"
    "Geometry.cpp"
    "MeshBoundary.cpp"
    "TetrahedralAabbHierarchy.cpp"
    "TriangleAabbHierarchy.cpp"
)