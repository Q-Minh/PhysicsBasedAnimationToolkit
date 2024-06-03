target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "HyperElasticity.h"
    "SaintVenantKirchhoffEnergy.h"
    "StableNeoHookeanEnergy.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "HyperElasticity.cpp"
    "SaintVenantKirchhoffEnergy.cpp"
    "StableNeoHookeanEnergy.cpp"
)