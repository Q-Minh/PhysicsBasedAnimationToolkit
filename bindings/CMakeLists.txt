pybind11_add_module(PhysicsBasedAnimationToolkit_Python
    MODULE
    EXCLUDE_FROM_ALL
)

set_target_properties(PhysicsBasedAnimationToolkit_Python
    PROPERTIES
    OUTPUT_NAME "pypbat"
    EXPORT_NAME "Python"
    FOLDER "PhysicsBasedAnimationToolkit/bindings"
)

target_sources(PhysicsBasedAnimationToolkit_Python
    PUBLIC
    FILE_SET api
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(PhysicsBasedAnimationToolkit_Python
    PRIVATE
    PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit
)

add_subdirectory(pypbat)

# Make source tree appear as on file system in IDEs
get_target_property(_pbat_python_sources PhysicsBasedAnimationToolkit_Python SOURCES)
get_target_property(_pbat_python_headers PhysicsBasedAnimationToolkit_Python HEADER_SET_api)
source_group(
    TREE ${CMAKE_CURRENT_SOURCE_DIR}
    PREFIX "source"
    FILES ${_pbat_python_sources}
)
source_group(
    TREE ${CMAKE_CURRENT_SOURCE_DIR}
    PREFIX "headers"
    FILES ${_pbat_python_headers}
)

# Make sure DLL dependencies end up together with python module
add_custom_command(
    TARGET PhysicsBasedAnimationToolkit_Python
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_RUNTIME_DLLS:PhysicsBasedAnimationToolkit_Python>
    $<TARGET_FILE_DIR:PhysicsBasedAnimationToolkit_Python>
    COMMAND_EXPAND_LISTS
)