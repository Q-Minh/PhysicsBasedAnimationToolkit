target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "AxisAlignedBoundingBox.h"
    "HashGrid.h"
    "HierarchicalHashGrid.h"
    "Geometry.h"
    "MeshBoundary.h"
    "TetrahedralAabbHierarchy.h"
    "TriangleAabbHierarchy.h"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "AxisAlignedBoundingBox.cpp"
    "HashGrid.cpp"
    "HierarchicalHashGrid.cpp"
    "Geometry.cpp"
    "MeshBoundary.cpp"
    "TetrahedralAabbHierarchy.cpp"
    "TriangleAabbHierarchy.cpp"
)