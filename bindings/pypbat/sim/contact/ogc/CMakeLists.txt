target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    FILES
    "ContactFace.h"
    "Input.h"
    "Ogc.h"
    "Params.h"
    "State.h"
)
target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "ContactFace.cpp"
    "Input.cpp"
    "Ogc.cpp"
    "Params.cpp"
    "State.cpp"
)