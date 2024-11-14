target_sources(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    "PythonBindings.cpp"
)

add_subdirectory(fem)
add_subdirectory(geometry)
add_subdirectory(graph)
add_subdirectory(gpu)
add_subdirectory(math)
add_subdirectory(profiling)
add_subdirectory(sim)

target_link_libraries(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    pybind11::headers
    Python::Module
)