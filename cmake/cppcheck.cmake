# cmake/cppcheck.cmake
# Provides a dedicated `cppcheck` CMake target that runs Cppcheck static analysis.
#
# Cppcheck is invoked with --project= pointing to either:
#   - compile_commands.json (Ninja/Makefile generators, requires CMAKE_EXPORT_COMPILE_COMMANDS=ON)
#   - The Visual Studio .sln file (Visual Studio generators)
#
# Usage: include(cppcheck) after the main library target has been defined.

find_program(CPPCHECK_EXECUTABLE NAMES cppcheck)

if(NOT CPPCHECK_EXECUTABLE)
    message(WARNING "PBAT -- cppcheck not found, the 'cppcheck' target will not be available. "
                    "Install cppcheck and make sure it is on your PATH.")
    return()
endif()

message(STATUS "PBAT -- Found cppcheck: ${CPPCHECK_EXECUTABLE}")

# Choose the project file that cppcheck should consume.
if(CMAKE_GENERATOR MATCHES "Visual Studio")
    # Visual Studio generators produce a .sln that cppcheck understands natively.
    set(_pbat_cppcheck_project "${CMAKE_BINARY_DIR}/${PROJECT_NAME}.sln")
elseif(CMAKE_EXPORT_COMPILE_COMMANDS)
    # Ninja / Makefile generators can produce compile_commands.json.
    set(_pbat_cppcheck_project "${CMAKE_BINARY_DIR}/compile_commands.json")
else()
    message(WARNING
        "PBAT -- Cannot determine a cppcheck project file. "
        "Either use a Visual Studio generator or set CMAKE_EXPORT_COMPILE_COMMANDS=ON "
        "(Ninja/Makefile generators). The 'cppcheck' target will not be available.")
    return()
endif()

message(STATUS "PBAT -- cppcheck project file: ${_pbat_cppcheck_project}")

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cppcheck)
add_custom_target(PhysicsBasedAnimationToolkit_Cppcheck
    COMMAND ${CPPCHECK_EXECUTABLE}
        --project=${_pbat_cppcheck_project}
        --cppcheck-build-dir=${CMAKE_CURRENT_BINARY_DIR}/cppcheck
        -i ${CMAKE_BINARY_DIR}
        --std=c++20
        --quiet
        # --template="{file}:{line}: [{severity}/{id}] {message}"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "PBAT -- Running cppcheck static analysis on ${_pbat_cppcheck_project}"
    VERBATIM
)
set_target_properties(PhysicsBasedAnimationToolkit_Cppcheck PROPERTIES FOLDER "PhysicsBasedAnimationToolkit/analysis")
