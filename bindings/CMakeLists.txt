Python_add_library(PhysicsBasedAnimationToolkit_Python MODULE WITH_SOABI)

set_target_properties(PhysicsBasedAnimationToolkit_Python
    PROPERTIES
    OUTPUT_NAME "_pbat"
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
    pybind11::headers
)

if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(PhysicsBasedAnimationToolkit_Python)
endif()

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

# NOTE: We assume that Python bindings link to statically built dependencies.
# Make sure DLL dependencies end up together with python module in build tree
# add_custom_command(
#     TARGET PhysicsBasedAnimationToolkit_Python
#     POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#     $<TARGET_RUNTIME_DLLS:PhysicsBasedAnimationToolkit_Python>
#     $<TARGET_FILE_DIR:PhysicsBasedAnimationToolkit_Python>
#     COMMAND_EXPAND_LISTS
# )

# Make sure Python module can find shared library dependencies (i.e. TBB) 
# in install tree on non-Windows platforms.
set_target_properties(PhysicsBasedAnimationToolkit_Python
    PROPERTIES
    INSTALL_RPATH "$ORIGIN"
)

# Install rules for scikit-build-core. Note that we only support building
# the python bindings with static linking to
# PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit. Otherwise, we
# need to handle installing the transitive DLL dependencies of our bindings.
include(GNUInstallDirs)
install(
    TARGETS PhysicsBasedAnimationToolkit_Python
    LIBRARY DESTINATION "pbatoolkit"
)

# NOTE: Our Python bindings only support static library dependencies, 
# however, TBB only comes as a dynamic library. We handle this special 
# case, unfortunately.
install(FILES $<TARGET_FILE:TBB::tbb> DESTINATION "pbatoolkit")
