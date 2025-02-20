if(PBAT_USE_CUDA)
    target_sources(PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
        PUBLIC
        FILE_SET api
        FILES
        "Aliases.h"
        "DisableWarnings.h"
        "Gpu.h"
    )

    add_subdirectory(impl)
    add_subdirectory(contact)
    add_subdirectory(common)
    add_subdirectory(geometry)
    add_subdirectory(profiling)
    add_subdirectory(xpbd)
    add_subdirectory(vbd)
endif()