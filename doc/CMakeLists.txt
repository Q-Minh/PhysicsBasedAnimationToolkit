find_package(Doxygen REQUIRED)
if (DOXYGEN_FOUND)
    set(DOXYGEN_PROJECT_BRIEF "Cross-platform C++20 library of algorithms and data structures commonly used in computer graphics research on physically-based simulation.")
    set(DOXYGEN_PROJECT_LOGO "img/pbat.svg")
    set(DOXYGEN_PROJECT_ICON "img/entei.icon.png")
    set(DOXYGEN_CREATE_SUBDIRS "YES")
    list(
        APPEND 
        DOXYGEN_EXTENSION_MAPPING 
        "cuh=C++" 
        "cu=C++"
    )
    list(
        APPEND
        DOXYGEN_EXCLUDE_SYMBOLS
        "detail::*"
        "*::detail::*"
        "test::*"
        "*::test::*"
        "internal::*"
        "*::internal::*"
    )
    set(DOXYGEN_SORT_GROUP_NAMES "YES")
    list(
        APPEND 
        DOXYGEN_FILE_PATTERNS 
        "*.h" 
        "*.hpp" 
        "*.cuh" 
        "*.cpp" 
        "*.cu" 
        "*.md" 
        "*.py"
    )
    set(DOXYGEN_GENERATE_HTML "YES")
    set(DOXYGEN_IMAGE_PATH "img")
    set(DOXYGEN_USE_MATHJAX "YES")
    set(DOXYGEN_MATHJAX_VERSION "MathJax_3")
    set(DOXYGEN_GENERATE_TREEVIEW "YES")
    set(DOXYGEN_DISABLE_INDEX "NO")
    set(DOXYGEN_FULL_SIDEBAR "NO")
    list(
        APPEND 
        DOXYGEN_HTML_EXTRA_STYLESHEET 
        "css/doxygen-awesome.css"
    )
    set(DOXYGEN_HTML_COLORSTYLE "LIGHT")
    set(DOXYGEN_HTML_HEADER "html/header.html")
    list(
        APPEND 
        DOXYGEN_HTML_EXTRA_FILES 
        "js/doxygen-awesome-darkmode-toggle.js"
        "js/doxygen-awesome-fragment-copy-button.js"
        "js/doxygen-awesome-paragraph-link.js"
        "js/doxygen-awesome-interactive-toc.js"
        "js/doxygen-awesome-tabs.js"
    )
    set(DOXYGEN_HTML_COPY_CLIPBOARD "NO")

    find_program(BIBTEX_EXECUTABLE bibtex)
    find_program(PERL_EXECUTABLE perl)
    if (BIBTEX_EXECUTABLE AND PERL_EXECUTABLE)
        list(
            APPEND 
            DOXYGEN_CITE_BIB_FILES
            "latex/bibliography.bib"
        )
    else()
        message(WARNING "PBAT[doc] -- BibTeX and Perl not found, bibliography will not be generated")
    endif()

    doxygen_add_docs(
        PhysicsBasedAnimationToolkit_Docs
        "${CMAKE_SOURCE_DIR}/source"
        "${CMAKE_SOURCE_DIR}/python/pbatoolkit"
        # List markdown files manually to ensure correct order
        "main.md"
        "features.md"
        "quickstart.md"
        "dependencies.md"
        "configuration.md"
        "build.md"
        "install.md"
        "profiling.md"
        "userguide.md"
        "gallery.md"
        ALL
        COMMENT "Generating docs"
    )

    # See https://cmake.org/cmake/help/latest/module/FindDoxygen.html#variable:DOXYGEN_OUTPUT_DIRECTORY
    install(
        DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html 
        DESTINATION share/doc
        COMPONENT Docs
        EXCLUDE_FROM_ALL
    )
else()
    message(FATAL_ERROR "Doxygen needs to be installed to generate the documentation")
endif()