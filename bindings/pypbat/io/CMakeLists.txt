target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "Archive.h"
    "Io.h"
)
target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "Archive.cpp"
    "Io.cpp"
)
