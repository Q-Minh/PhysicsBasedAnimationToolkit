target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "PythonBindings.cpp"
)

add_subdirectory(fem)
add_subdirectory(profiling)