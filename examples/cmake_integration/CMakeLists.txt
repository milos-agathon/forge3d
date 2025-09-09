# Example CMake project showing forge3d integration
cmake_minimum_required(VERSION 3.20)
project(forge3d_example VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to use forge3d as subdirectory vs. installed package
option(USE_FORGE3D_SUBDIR "Use forge3d as subdirectory instead of finding installed package" ON)

if(USE_FORGE3D_SUBDIR)
    # Method 1: Add forge3d as subdirectory
    # Assumes forge3d is in parent directory or adjust path as needed
    set(FORGE3D_BUILD_EXAMPLES OFF CACHE BOOL "Disable forge3d examples" FORCE)
    set(FORGE3D_BUILD_TESTS OFF CACHE BOOL "Disable forge3d tests" FORCE)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../ forge3d)
    
    message(STATUS "Using forge3d as subdirectory")
else()
    # Method 2: Find installed forge3d package
    find_package(forge3d REQUIRED)
    message(STATUS "Using installed forge3d package")
endif()

# Find Python for running examples
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Create a target that runs Python examples
add_custom_target(run_python_examples
    COMMAND ${Python3_EXECUTABLE} -c "
import sys, os
sys.path.insert(0, '${CMAKE_CURRENT_SOURCE_DIR}/../../python')
print('Testing forge3d import...')
try:
    import forge3d
    print(f'forge3d version: {forge3d.__version__}')
    print('forge3d imported successfully!')
    
    # Run a simple test
    print('Running simple rendering test...')
    renderer = forge3d.Renderer(512, 512)
    rgba = renderer.render_triangle_png()
    print(f'Rendered triangle: {len(rgba)} bytes')
    print('Example completed successfully!')
    
except ImportError as e:
    print(f'Failed to import forge3d: {e}')
    sys.exit(1)
except Exception as e:
    print(f'Error running example: {e}')
    sys.exit(1)
"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running Python examples with forge3d"
    VERBATIM
)

# Make sure forge3d is built before running examples
if(USE_FORGE3D_SUBDIR)
    add_dependencies(run_python_examples forge3d_rust_build)
endif()

# Create a test that verifies the integration
enable_testing()

add_test(
    NAME forge3d_import_test
    COMMAND ${Python3_EXECUTABLE} -c "
import sys, os
sys.path.insert(0, '${CMAKE_CURRENT_SOURCE_DIR}/../../python')
import forge3d
print('forge3d import test passed')
"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Simple C++ example that shows how you might integrate with forge3d
# (Note: forge3d is primarily a Python library, this is just a conceptual example)
add_executable(cpp_example cpp_example.cpp)

# If you wanted to call Python from C++, you'd link against Python
target_link_libraries(cpp_example ${Python3_LIBRARIES})
target_include_directories(cpp_example PRIVATE ${Python3_INCLUDE_DIRS})

# Installation
install(TARGETS cpp_example
    RUNTIME DESTINATION bin
)

# Package configuration
set(CPACK_PACKAGE_NAME "forge3d-example")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Example project using forge3d")
include(CPack)

# Print configuration summary
message(STATUS "")
message(STATUS "forge3d Example Configuration:")
message(STATUS "  Project version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Python executable: ${Python3_EXECUTABLE}")
message(STATUS "  Use forge3d subdirectory: ${USE_FORGE3D_SUBDIR}")
message(STATUS "")
message(STATUS "Available targets:")
message(STATUS "  cpp_example        - Simple C++ example")
message(STATUS "  run_python_examples - Run Python examples")
message(STATUS "")
message(STATUS "To run examples:")
message(STATUS "  cmake --build build --target run_python_examples")
message(STATUS "")