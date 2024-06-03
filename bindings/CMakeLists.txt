pybind11_add_module(PhysicsBasedAnimationToolkit_Python
    MODULE
    EXCLUDE_FROM_ALL
    "src/PythonBindings.cpp")

target_link_libraries(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
)