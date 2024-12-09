target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PUBLIC
    FILE_SET api
    FILES
    "Cube.h"
    "Enums.h"
    "Model.h"
)
target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
    PRIVATE
    "Cube.cpp"
)

if(PBAT_PRECOMPILE_LARGE_MODELS)
    target_compile_definitions(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit PUBLIC PBAT_WITH_PRECOMPILED_LARGE_MODELS)
    target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
        PUBLIC
        FILE_SET implementation
        FILES
        "Armadillo.h"
    )
    target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
        PRIVATE
        "Armadillo.cpp"
    )
endif()