// Example C++ program showing how to integrate with forge3d
// Note: forge3d is primarily a Python library, so this example shows
// how to call Python from C++ to use forge3d functionality

#include <Python.h>
#include <iostream>
#include <string>
#include <vector>

class Forge3DWrapper {
private:
    PyObject* forge3d_module;
    PyObject* renderer_class;

public:
    Forge3DWrapper() : forge3d_module(nullptr), renderer_class(nullptr) {
        // Initialize Python interpreter
        Py_Initialize();
        
        if (!Py_IsInitialized()) {
            throw std::runtime_error("Failed to initialize Python");
        }
        
        // Add forge3d to Python path (adjust path as needed)
        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.insert(0, '../../python')");
        
        // Import forge3d module
        forge3d_module = PyImport_ImportModule("forge3d");
        if (!forge3d_module) {
            PyErr_Print();
            throw std::runtime_error("Failed to import forge3d module");
        }
        
        // Get Renderer class
        renderer_class = PyObject_GetAttrString(forge3d_module, "Renderer");
        if (!renderer_class) {
            PyErr_Print();
            throw std::runtime_error("Failed to get Renderer class");
        }
    }
    
    ~Forge3DWrapper() {
        Py_XDECREF(renderer_class);
        Py_XDECREF(forge3d_module);
        Py_Finalize();
    }
    
    std::string get_version() {
        PyObject* version_attr = PyObject_GetAttrString(forge3d_module, "__version__");
        if (!version_attr) {
            PyErr_Print();
            return "unknown";
        }
        
        const char* version_str = PyUnicode_AsUTF8(version_attr);
        std::string result(version_str);
        Py_DECREF(version_attr);
        return result;
    }
    
    bool render_triangle_test() {
        // Create Renderer instance (512x512)
        PyObject* args = PyTuple_Pack(2, PyLong_FromLong(512), PyLong_FromLong(512));
        PyObject* renderer = PyObject_CallObject(renderer_class, args);
        Py_DECREF(args);
        
        if (!renderer) {
            PyErr_Print();
            return false;
        }
        
        // Call render_triangle_png method
        PyObject* result = PyObject_CallMethod(renderer, "render_triangle_png", nullptr);
        if (!result) {
            PyErr_Print();
            Py_DECREF(renderer);
            return false;
        }
        
        // Check if result is bytes
        if (PyBytes_Check(result)) {
            Py_ssize_t size = PyBytes_Size(result);
            std::cout << "Successfully rendered triangle: " << size << " bytes" << std::endl;
        } else {
            std::cout << "Unexpected return type from render_triangle_png" << std::endl;
        }
        
        Py_DECREF(result);
        Py_DECREF(renderer);
        return true;
    }
};

int main() {
    std::cout << "forge3d C++ Integration Example" << std::endl;
    std::cout << "================================" << std::endl;
    
    try {
        Forge3DWrapper forge3d;
        
        // Get version information
        std::string version = forge3d.get_version();
        std::cout << "forge3d version: " << version << std::endl;
        
        // Run a simple rendering test
        std::cout << "\nRunning triangle rendering test..." << std::endl;
        bool success = forge3d.render_triangle_test();
        
        if (success) {
            std::cout << "✓ Triangle rendering test passed!" << std::endl;
        } else {
            std::cout << "✗ Triangle rendering test failed!" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\nExample completed successfully!" << std::endl;
    return 0;
}

/* 
Alternative approach: Using pybind11 for easier Python-C++ integration

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

namespace py = pybind11;

int main() {
    py::scoped_interpreter guard{};
    
    try {
        py::module_ forge3d = py::module_::import("forge3d");
        py::object renderer_class = forge3d.attr("Renderer");
        py::object renderer = renderer_class(512, 512);
        
        py::bytes result = renderer.attr("render_triangle_png")();
        std::cout << "Rendered " << result.size() << " bytes" << std::endl;
        
    } catch (const py::error_already_set& e) {
        std::cerr << "Python error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
*/