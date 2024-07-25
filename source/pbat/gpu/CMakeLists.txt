if(PBAT_USE_CUDA)
    target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
        PUBLIC
        FILE_SET api
        FILES
        "Aliases.h"
    )

    add_subdirectory(common)
    add_subdirectory(geometry)
endif()