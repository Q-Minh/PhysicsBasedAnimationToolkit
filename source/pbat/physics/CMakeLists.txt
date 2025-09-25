target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "Enums.h"
    "HyperElasticity.h"
    "Physics.h"
    "SaintVenantKirchhoffEnergy.h"
    "StableNeoHookeanEnergy.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Enums.cpp"
    "HyperElasticity.cpp"
    "SaintVenantKirchhoffEnergy.cpp"
    "StableNeoHookeanEnergy.cpp"
)