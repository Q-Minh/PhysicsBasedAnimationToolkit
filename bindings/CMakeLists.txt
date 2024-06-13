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

# Make sure Python module can find shared library dependencies (i.e. TBB)
# in install tree on non-Windows platforms.
include(GNUInstallDirs)
list(APPEND _pbat_python_rpaths
    "$ORIGIN"
    "$ORIGIN/${CMAKE_INSTALL_LIBDIR}"
    "$ORIGIN/${CMAKE_INSTALL_BINDIR}"
)
set_target_properties(PhysicsBasedAnimationToolkit_Python
    PROPERTIES
    INSTALL_RPATH ${_pbat_python_rpaths}
)

# Install rules for scikit-build-core. Note that we only support building
# the python bindings with static linking to
# PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit. Otherwise, we
# need to handle installing the transitive DLL dependencies of our bindings.
install(
    TARGETS PhysicsBasedAnimationToolkit_Python
    LIBRARY DESTINATION "pbatoolkit"
)

# NOTE: Install shared dependencies
foreach(_pbat_dependency IN ITEMS TBB::tbb fmt::fmt)
    install(FILES $<TARGET_FILE:${_pbat_dependency}> DESTINATION "pbatoolkit")
endforeach()
