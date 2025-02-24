if (MSVC)
    set(
        CMAKE_CXX_FLAGS 
        "${CMAKE_CXX_FLAGS} /MP /bigobj /utf-8" 
    )
endif()