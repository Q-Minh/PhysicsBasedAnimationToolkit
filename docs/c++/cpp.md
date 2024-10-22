Here's a sample content for the `quick_start/cpp.md` file to help users get started with the C++ version of the Physics Based Animation Toolkit:

```markdown
# Quick Start Guide for C++

This guide will walk you through setting up the **Physics Based Animation Toolkit** for use in C++ projects. By the end of this guide, you'll be able to build and run basic simulations using the toolkit in a C++ environment.

## Prerequisites

Before starting, ensure that you have the following installed:

- **CMake** (version 3.15 or higher)
- **C++17** or higher compatible compiler (GCC, Clang, MSVC, etc.)
- **vcpkg** (for dependency management)
- **CUDA Toolkit** (for GPU support, optional)
- **Python 3.x** (if you plan to use the Python bindings alongside C++)

## Step 1: Clone the Repository

First, clone the Physics Based Animation Toolkit from the official GitHub repository:

```bash
git clone https://github.com/Q-Minh/PhysicsBasedAnimationToolkit.git
cd PhysicsBasedAnimationToolkit
```

## Step 2: Set Up Dependencies

We use **vcpkg** to manage external dependencies like Eigen and other libraries. Make sure that vcpkg is initialized and the required packages are installed.

### Install Dependencies using vcpkg

If you haven't already set up **vcpkg**, follow the instructions below:

```bash
# Clone vcpkg (if not installed)
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg

# Bootstrap and integrate with your system
./bootstrap-vcpkg.sh  # For Linux/macOS
.\bootstrap-vcpkg.bat  # For Windows

# Integrate with your environment
./vcpkg integrate install

# Install required libraries
./vcpkg install eigen3 jsoncpp fmt
```

The dependencies will now be available to your project when building with CMake.

## Step 3: Build the Project

You can now build the toolkit with CMake:

```bash
mkdir build
cd build

# Configure the project with CMake
cmake .. -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake

# Build the project
cmake --build . --config Release
```

This will build the core C++ libraries of the Physics Based Animation Toolkit.

## Step 4: Run an Example

After building, you can run one of the provided examples to verify the setup.

### Run a Simple Simulation

The toolkit comes with example C++ code that demonstrates how to set up a simple physics simulation:

```bash
./bin/simple_simulation
```

This will execute a basic example simulation. You can modify and extend this example to suit your needs.

## Step 5: Integrating with Your Own C++ Project

To use the toolkit in your own project, link it as a library. Below is an example of how you can set this up in your CMake project:

### CMake Configuration

Add the following to your `CMakeLists.txt`:

```cmake
# Add the Physics Based Animation Toolkit to your project
add_subdirectory(path/to/PhysicsBasedAnimationToolkit)

# Link the library
target_link_libraries(your_project_name PRIVATE pbatoolkit)
```

You can now include and use the functionality provided by the Physics Based Animation Toolkit in your project.

## Step 6: GPU Support (Optional)

If you want to enable GPU support, ensure you have **CUDA Toolkit** installed and available in your environment. GPU features can be enabled by passing the following argument to CMake during configuration:

```bash
cmake .. -DUSE_CUDA=ON
```

This will compile the GPU-specific parts of the toolkit.

## Additional Resources

- [Full API Documentation](../docs/index.md)
- [Examples and Tutorials](../examples/index.md)
- [Contributing Guide](../contributing.md)

For more detailed examples and usage, please refer to the **Examples** section in the documentation.

---

Happy coding with the Physics Based Animation Toolkit!
```

This file provides a basic quick-start guide for using the Physics Based Animation Toolkit with C++, including installation steps, building the project, and running simple simulations. Feel free to customize the content as needed for your specific setup!