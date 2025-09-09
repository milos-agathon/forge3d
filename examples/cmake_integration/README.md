# CMake Integration Example

This directory demonstrates how to integrate forge3d into a CMake-based project.

## Overview

This example shows two approaches:
1. **Subdirectory integration** - Adding forge3d as a subdirectory
2. **Package integration** - Using forge3d as an installed CMake package

## Quick Start

### Prerequisites

- CMake 3.20+
- Rust toolchain (rustc, cargo)
- Python 3.8+ with development headers
- C++ compiler (for the C++ example)

### Building

```bash
# From the examples/cmake_integration directory
mkdir build && cd build

# Configure (using forge3d as subdirectory)
cmake .. -DUSE_FORGE3D_SUBDIR=ON

# Build
cmake --build .

# Run Python examples
cmake --build . --target run_python_examples

# Run C++ example
./cpp_example  # Linux/macOS
# or
cpp_example.exe  # Windows
```

### Using Installed forge3d Package

```bash
# First, install forge3d system-wide
cd ../../..  # Back to forge3d root
cmake -B build -S .
cmake --build build
sudo cmake --install build

# Then configure example to use installed package
cd examples/cmake_integration
mkdir build && cd build
cmake .. -DUSE_FORGE3D_SUBDIR=OFF
cmake --build .
```

## Project Structure

```
cmake_integration/
├── CMakeLists.txt          # Main CMake configuration
├── cpp_example.cpp         # C++ integration example
├── README.md              # This file
└── build/                 # Build directory (created)
```

## Features Demonstrated

### CMake Features
- ✅ **Subdirectory integration** with forge3d
- ✅ **Package finding** for installed forge3d
- ✅ **Cross-platform builds** (Windows, macOS, Linux)
- ✅ **Python integration** from CMake
- ✅ **Custom targets** for running examples
- ✅ **Testing integration** with CTest

### forge3d Integration
- ✅ **Python module usage** from CMake
- ✅ **C++ to Python binding** (using Python C API)
- ✅ **Rust library building** via Cargo
- ✅ **Development workflow** integration

## Detailed Usage

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_FORGE3D_SUBDIR` | `ON` | Use forge3d as subdirectory vs. installed package |
| `CMAKE_BUILD_TYPE` | `Release` | Build configuration |

### Available Targets

| Target | Description |
|--------|-------------|
| `cpp_example` | C++ example executable |
| `run_python_examples` | Run Python examples using forge3d |

### Testing

```bash
# Run all tests
ctest --verbose

# Run specific test
ctest -R forge3d_import_test --verbose
```

## C++ Integration Details

The `cpp_example.cpp` demonstrates how to use forge3d from C++:

1. **Python Embedding** - Initializes Python interpreter
2. **Module Import** - Imports forge3d Python module
3. **API Calls** - Calls forge3d functions from C++
4. **Error Handling** - Proper Python error handling

### Key Code Patterns

```cpp
// Initialize Python and import forge3d
Py_Initialize();
PyObject* module = PyImport_ImportModule("forge3d");

// Create renderer instance
PyObject* renderer = PyObject_CallObject(renderer_class, args);

// Call methods
PyObject* result = PyObject_CallMethod(renderer, "render_triangle_png", nullptr);
```

### Alternative: pybind11

For easier C++/Python integration, consider pybind11 (commented code in example):

```cpp
#include <pybind11/embed.h>
namespace py = pybind11;

py::module_ forge3d = py::module_::import("forge3d");
py::object renderer = forge3d.attr("Renderer")(512, 512);
py::bytes result = renderer.attr("render_triangle_png")();
```

## Troubleshooting

### Common Issues

**forge3d not found:**
```
ImportError: No module named 'forge3d'
```
**Solution:** Ensure forge3d is built and Python path is correct.

**Rust not found:**
```
CMake Error: Could not find program: cargo
```
**Solution:** Install Rust from https://rustup.rs/

**Python headers missing:**
```
fatal error: Python.h: No such file or directory
```
**Solution:** Install Python development packages:
- Ubuntu/Debian: `sudo apt install python3-dev`
- CentOS/RHEL: `sudo yum install python3-devel`
- macOS: `brew install python`
- Windows: Python from python.org includes headers

**Build fails on Windows:**
```
error LNK2019: unresolved external symbol
```
**Solution:** Ensure correct Python library linking:
```cmake
target_link_libraries(cpp_example ${Python3_LIBRARIES})
```

### Debug Information

Enable verbose CMake output:
```bash
cmake --build . --verbose
```

Check Python configuration:
```bash
python3-config --includes --libs
```

## Performance Considerations

### Build Time
- Use `CMAKE_BUILD_TYPE=Release` for optimized builds
- Enable parallel builds: `cmake --build . --parallel`
- Consider `ccache` for faster rebuilds

### Runtime Performance
- Minimize Python interpreter startup/shutdown
- Reuse Python objects when possible
- Consider pybind11 for better performance

## Integration Patterns

### Pattern 1: Embedded Python
```cpp
// Good for: Simple use cases, full control
Py_Initialize();
// Use Python C API directly
Py_Finalize();
```

### Pattern 2: pybind11
```cpp
// Good for: Modern C++, easier syntax
py::scoped_interpreter guard{};
py::module_ forge3d = py::module_::import("forge3d");
```

### Pattern 3: Subprocess
```cpp
// Good for: Isolation, simple integration
system("python -c 'import forge3d; ...'");
```

## Extending the Example

To add more functionality:

1. **More APIs** - Call additional forge3d functions
2. **Error Handling** - Improve Python error handling
3. **Threading** - Use forge3d in multi-threaded context
4. **Configuration** - Pass parameters from C++ to Python
5. **Packaging** - Create installable packages

## References

- [Python C API Documentation](https://docs.python.org/3/c-api/)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)
- [CMake Documentation](https://cmake.org/documentation/)
- [forge3d API Documentation](../../docs/)