target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "PythonBindings.cpp"
)

add_subdirectory(fem)
add_subdirectory(profiling)

target_link_libraries(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    PhysicsBasedAnimationToolkit_Python_Fem
    Python::Module
    pybind11::headers
)