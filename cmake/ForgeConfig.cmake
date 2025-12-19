# ForgeConfig.cmake - CMake configuration for forge3d integration

# Guard against multiple inclusion
if(FORGE3D_CONFIG_INCLUDED)
    return()
endif()
set(FORGE3D_CONFIG_INCLUDED TRUE)

# Version information
set(FORGE3D_VERSION_MAJOR 1)
set(FORGE3D_VERSION_MINOR 2)
set(FORGE3D_VERSION_PATCH 0)
set(FORGE3D_VERSION ${FORGE3D_VERSION_MAJOR}.${FORGE3D_VERSION_MINOR}.${FORGE3D_VERSION_PATCH})

# Find package components
include(CMakeFindDependencyMacro)

# Required dependencies
find_dependency(Python3 COMPONENTS Interpreter Development)

# Optional: Find Rust toolchain
find_program(CARGO_EXECUTABLE cargo)
find_program(RUSTC_EXECUTABLE rustc)

if(NOT CARGO_EXECUTABLE OR NOT RUSTC_EXECUTABLE)
    message(WARNING "Rust toolchain not found. forge3d requires Rust for building.")
    set(FORGE3D_RUST_AVAILABLE FALSE)
else()
    set(FORGE3D_RUST_AVAILABLE TRUE)
    
    # Get Rust version
    execute_process(
        COMMAND ${RUSTC_EXECUTABLE} --version
        OUTPUT_VARIABLE FORGE3D_RUST_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    message(STATUS "Found Rust: ${FORGE3D_RUST_VERSION}")
endif()

# Utility functions for forge3d integration
function(forge3d_add_python_module target_name)
    cmake_parse_arguments(FORGE3D 
        ""
        "MODULE_NAME;PYTHON_SOURCE_DIR"
        "RUST_FEATURES;EXTRA_ARGS"
        ${ARGN}
    )
    
    if(NOT FORGE3D_MODULE_NAME)
        set(FORGE3D_MODULE_NAME ${target_name})
    endif()
    
    if(NOT FORGE3D_PYTHON_SOURCE_DIR)
        set(FORGE3D_PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python")
    endif()
    
    # Build Rust library
    set(build_args --lib)
    
    if(FORGE3D_RUST_FEATURES)
        string(REPLACE ";" "," features_str "${FORGE3D_RUST_FEATURES}")
        list(APPEND build_args --features ${features_str})
    endif()
    
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        list(APPEND build_args --release)
        set(profile_dir "release")
    else()
        set(profile_dir "debug")
    endif()
    
    if(FORGE3D_EXTRA_ARGS)
        list(APPEND build_args ${FORGE3D_EXTRA_ARGS})
    endif()
    
    # Determine output paths
    set(target_dir "${CMAKE_CURRENT_BINARY_DIR}/target/${profile_dir}")
    
    if(WIN32)
        set(lib_extension ".dll")
        set(lib_prefix "")
    else()
        set(lib_extension ".so")
        set(lib_prefix "lib")
    endif()
    
    set(rust_lib "${target_dir}/${lib_prefix}${FORGE3D_MODULE_NAME}${lib_extension}")
    
    # Custom command to build Rust library
    add_custom_command(
        OUTPUT ${rust_lib}
        COMMAND ${CMAKE_COMMAND} -E env
                "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/target"
                ${CARGO_EXECUTABLE} build ${build_args}
                --target-dir "${CMAKE_CURRENT_BINARY_DIR}/target"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Building ${target_name} Rust library"
        VERBATIM
    )
    
    # Create CMake target
    add_custom_target(${target_name} ALL
        DEPENDS ${rust_lib}
    )
    
    # Set target properties
    set_target_properties(${target_name} PROPERTIES
        FORGE3D_RUST_LIBRARY ${rust_lib}
        FORGE3D_MODULE_NAME ${FORGE3D_MODULE_NAME}
        FORGE3D_PYTHON_SOURCE_DIR ${FORGE3D_PYTHON_SOURCE_DIR}
    )
    
endfunction()

# Function to install forge3d Python module
function(forge3d_install_python_module target_name)
    cmake_parse_arguments(FORGE3D
        ""
        "DESTINATION;MODULE_NAME"
        ""
        ${ARGN}
    )
    
    get_target_property(rust_lib ${target_name} FORGE3D_RUST_LIBRARY)
    get_target_property(module_name ${target_name} FORGE3D_MODULE_NAME)
    get_target_property(python_source_dir ${target_name} FORGE3D_PYTHON_SOURCE_DIR)
    
    if(NOT FORGE3D_DESTINATION)
        # Auto-detect Python site-packages
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('purelib'))"
            OUTPUT_VARIABLE FORGE3D_DESTINATION
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()
    
    if(NOT FORGE3D_MODULE_NAME)
        set(FORGE3D_MODULE_NAME ${module_name})
    endif()
    
    # Install Python source files
    if(EXISTS ${python_source_dir})
        install(
            DIRECTORY ${python_source_dir}/
            DESTINATION ${FORGE3D_DESTINATION}/
            FILES_MATCHING
            PATTERN "*.py"
            PATTERN "*.pyi"
        )
    endif()
    
    # Install Rust library
    if(WIN32)
        set(ext_suffix ".pyd")
    else()
        set(ext_suffix ".so")
    endif()
    
    install(
        FILES ${rust_lib}
        DESTINATION ${FORGE3D_DESTINATION}/${FORGE3D_MODULE_NAME}/
        RENAME "_${FORGE3D_MODULE_NAME}${ext_suffix}"
    )
    
endfunction()

# Function to setup forge3d development environment
function(forge3d_setup_development)
    cmake_parse_arguments(FORGE3D
        "ENABLE_CLIPPY;ENABLE_TESTS;ENABLE_DOCS"
        ""
        ""
        ${ARGN}
    )
    
    # Format target
    add_custom_target(rust_fmt
        COMMAND ${CARGO_EXECUTABLE} fmt
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Formatting Rust code"
    )
    
    # Clippy target
    if(FORGE3D_ENABLE_CLIPPY)
        add_custom_target(rust_clippy
            COMMAND ${CARGO_EXECUTABLE} clippy -- -D warnings
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Running Clippy linter"
        )
    endif()
    
    # Test target
    if(FORGE3D_ENABLE_TESTS)
        add_custom_target(rust_test
            COMMAND ${CARGO_EXECUTABLE} test
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Running Rust tests"
        )
    endif()
    
    # Documentation target
    if(FORGE3D_ENABLE_DOCS)
        add_custom_target(rust_docs
            COMMAND ${CARGO_EXECUTABLE} doc --no-deps --open
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generating Rust documentation"
        )
    endif()
    
endfunction()

# Export variables for use by consumers
set(FORGE3D_FOUND TRUE)
set(FORGE3D_INCLUDE_DIRS "")  # No C++ headers for this project
set(FORGE3D_LIBRARIES "")    # Python extension, no linkable libraries

message(STATUS "forge3d configuration loaded (version ${FORGE3D_VERSION})")

# Validate configuration
if(NOT FORGE3D_RUST_AVAILABLE)
    message(FATAL_ERROR "forge3d requires Rust toolchain (cargo and rustc)")
endif()

if(NOT Python3_FOUND)
    message(FATAL_ERROR "forge3d requires Python 3 development headers")
endif()

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "No build type specified, defaulting to Release")
endif()