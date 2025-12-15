---
sidebar_position: 2.3
title: "Building ROS 2 Packages"
---

# Building ROS 2 Packages

## Overview

This chapter covers the process of creating, building, and managing ROS 2 packages. A ROS 2 package is the fundamental unit of organization in the ROS 2 ecosystem, containing source code, configuration files, launch files, and documentation. Understanding how to properly structure and build packages is essential for developing humanoid robot applications.

## Package Structure and Organization

### Standard Package Layout

A well-structured ROS 2 package follows a standard layout:

```
my_robot_package/
├── CMakeLists.txt              # Build configuration for C++
├── package.xml                 # Package metadata and dependencies
├── src/                        # Source code files
│   ├── my_node.cpp
│   └── my_library.cpp
├── include/                    # Header files (C++)
│   └── my_robot_package/
│       └── my_header.hpp
├── launch/                     # Launch files for starting nodes
│   ├── my_launch.py
│   └── my_launch.xml
├── config/                     # Configuration files
│   ├── parameters.yaml
│   └── robot_description.urdf
├── msg/                        # Custom message definitions
│   ├── MyMessage.msg
│   └── AnotherMessage.msg
├── srv/                        # Custom service definitions
│   └── MyService.srv
├── action/                     # Custom action definitions
│   └── MyAction.action
├── test/                       # Unit and integration tests
│   ├── test_my_node.cpp
│   └── test_my_node.py
├── scripts/                    # Standalone scripts
│   └── utility_script.py
└── README.md                   # Package documentation
```

### Package Metadata (package.xml)

The `package.xml` file contains metadata about the package:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>1.0.0</version>
  <description>Package for my robot functionality</description>
  <maintainer email="maintainer@todo.todo">Maintainer Name</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

## Build System: ament

### ament Build System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    ament Build System                   │
│                                                         │
│  ament_cmake ──► CMake ──► Compiler ──► Executables    │
│      │             │         │            │             │
│      │             │         │            ▼             │
│      │             │         └──► Libraries             │
│      │             │                                    │
│  ament_python ───► Python ──────────────────────────────┘
│                     │
│                     ▼
│              Python Packages
└─────────────────────────────────────────────────────────┘
```

### CMakeLists.txt Structure

A typical `CMakeLists.txt` for a C++ ROS 2 package:

```cmake
cmake_minimum_required(VERSION 3.8)
project(my_robot_package)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Add executable
add_executable(my_node
  src/my_node.cpp
)

# Link libraries
ament_target_dependencies(my_node
  rclcpp
  std_msgs
  sensor_msgs
)

# Install executables
install(TARGETS
  my_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install other files
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
```

## Creating Packages

### Using ros2 pkg Command

The `ros2 pkg create` command provides a convenient way to create new packages:

```bash
# Create a C++ package
ros2 pkg create --build-type ament_cmake --dependencies rclcpp std_msgs my_cpp_package

# Create a Python package
ros2 pkg create --build-type ament_python --dependencies rclpy std_msgs my_python_package

# Create a package with custom options
ros2 pkg create --build-type ament_cmake \
                --dependencies rclcpp std_msgs sensor_msgs \
                --maintainer-email "your@email.com" \
                --maintainer-name "Your Name" \
                --license "Apache-2.0" \
                my_robot_control
```

### Package Creation Workflow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Define        │    │   Create        │    │   Implement     │
│   Requirements  │───▶│   Package       │───▶│   Functionality │
│   & Dependencies│    │   Structure     │    │   & Features    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Design        │    │   Build         │    │   Test &        │
│   Architecture  │    │   & Compile     │    │   Debug         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   Package Ready for     │
                    │   Integration & Use     │
                    └─────────────────────────┘
```

## Building Packages

### Build Commands

**Standard Build:**
```bash
# Build a specific package
colcon build --packages-select my_robot_package

# Build with specific compiler options
colcon build --packages-select my_robot_package --cmake-args -DCMAKE_BUILD_TYPE=Release

# Build with parallel jobs
colcon build --packages-select my_robot_package --parallel-workers 4
```

**Advanced Build Options:**
```bash
# Build only C++ packages
colcon build --packages-select my_robot_package --packages-skip-by-dep python

# Build with symlinks (faster rebuilds)
colcon build --packages-select my_robot_package --symlink-install

# Build and run tests
colcon build --packages-select my_robot_package --cmake-args -DBUILD_TESTING=ON
```

### Build Process Visualization

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Source        │    │   Compilation   │    │   Linking &     │
│   Code          │───▶│   & Processing  │───▶│   Installation  │
│   (.cpp, .py)   │    │   (CMake, etc.) │    │   (Binaries,    │
└─────────────────┘    └─────────────────┘    │   Libraries)    │
         │                       │             └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐              │
         │              │   Dependencies  │              │
         │              │   Resolution    │              │
         │              │   & Download    │              │
         │              └─────────────────┘              │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   Build Artifacts       │
                    │   (Executables,         │
                    │   Libraries, Scripts)   │
                    └─────────────────────────┘
```

## Package Dependencies

### Dependency Types

**Build Dependencies:**
- Required during compilation
- Listed in `package.xml` as `<build_depend>`
- Examples: header files, build tools

**Execution Dependencies:**
- Required during runtime
- Listed in `package.xml` as `<exec_depend>`
- Examples: runtime libraries, launch files

**Test Dependencies:**
- Required for testing
- Listed in `package.xml` as `<test_depend>`
- Examples: testing frameworks, linters

### Dependency Management

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   my_robot_pkg  │    │   sensor_pkg    │    │   control_pkg   │
│   (depends on)  │───▶│   (provides)    │    │   (provides)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   perception_   │    │   camera_driver │    │   joint_control │
│   node          │    │   node          │    │   node          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   my_robot_system       │
                    │   (Integrated System)   │
                    └─────────────────────────┘
```

## Cross-Compilation for Humanoid Robots

### Target Hardware Considerations

When building for humanoid robot hardware like NVIDIA Jetson:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Development   │    │   Cross-        │    │   Target        │
│   Machine       │───▶│   Compilation   │───▶│   Hardware      │
│   (x86_64)      │    │   (aarch64)     │    │   (ARM)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐              │
         │              │   Build System  │              │
         │              │   (colcon,      │              │
         │              │   cross-comp)   │              │
         │              └─────────────────┘              │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   Optimized Binaries    │
                    │   for Target Platform   │
                    └─────────────────────────┘
```

### Cross-Compilation Setup

```bash
# Create a cross-compilation workspace
mkdir -p cross_ws/src
cd cross_ws/src

# Clone packages
git clone https://github.com/robotpkg/my_robot_pkg.git

# Build for target architecture
colcon build --build-base build_aarch64 \
             --install-base install_aarch64 \
             --cmake-args -DCMAKE_TOOLCHAIN_FILE=../toolchain.cmake
```

## Testing and Quality Assurance

### Unit Testing Structure

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Source Code   │    │   Test Code     │    │   Test Runner   │
│   (src/)        │    │   (test/)       │    │   (ament_test)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Component     │    │   Test Cases    │    │   Test Results  │
│   Under Test    │    │   (GTest, PyTest)│   │   (JUnit XML)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   Code Coverage &       │
                    │   Quality Metrics       │
                    └─────────────────────────┘
```

### CMakeLists.txt for Testing

```cmake
# Enable testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)

  # Linting
  ament_lint_auto_find_test_dependencies()

  # C++ tests
  ament_add_gtest(test_my_node
    test/test_my_node.cpp
  )
  target_link_libraries(test_my_node
    my_library
  )

  # Python tests
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_my_node_py
    test/test_my_node.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
endif()
```

## Performance Optimization

### Build Optimization Flags

**Release Build with Optimizations:**
```cmake
# In CMakeLists.txt
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG")
```

**Profile-Guided Optimization:**
```bash
# Build with profiling
colcon build --packages-select my_robot_package \
             --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo \
             -DCMAKE_CXX_FLAGS="-fprofile-generate"

# Run application to generate profile data
ros2 run my_robot_package my_node

# Rebuild with optimization based on profile
colcon build --packages-select my_robot_package \
             --cmake-args -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_CXX_FLAGS="-fprofile-use -fprofile-correction"
```

### Memory and Resource Optimization

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Source Code   │    │   Optimized     │    │   Resource      │
│   (Unoptimized) │───▶│   Build         │───▶│   Efficient     │
│                 │    │   (Optimized)   │    │   Execution     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐              │
         │              │   Compiler      │              │
         │              │   Optimizations │              │
         │              │   (-O3, -march) │              │
         │              └─────────────────┘              │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   Optimized Binaries    │
                    │   for Robot Hardware    │
                    └─────────────────────────┘
```

## Best Practices

### Package Organization Guidelines

1. **Single Responsibility**: Each package should have a clear, focused purpose
2. **Logical Grouping**: Group related functionality in the same package
3. **Dependency Management**: Keep dependencies minimal and well-defined
4. **Documentation**: Include README, API docs, and usage examples
5. **Testing**: Include comprehensive unit and integration tests

### Build Optimization Guidelines

1. **Use Appropriate Build Types**: Debug for development, Release for deployment
2. **Optimize for Target Hardware**: Use architecture-specific optimizations
3. **Profile Before Optimizing**: Measure performance before making changes
4. **Incremental Builds**: Use build caching and symlinks for faster iterations
5. **Continuous Integration**: Automate building and testing

## Troubleshooting Common Issues

### Build Errors

**Common CMake Issues:**
- Missing dependencies in `package.xml`
- Incorrect include paths
- Missing library links
- Version compatibility issues

**Resolution Strategies:**
```bash
# Clean build
rm -rf build/ install/ log/
colcon build

# Verbose build for debugging
colcon build --event-handlers console_direct+

# Build specific package with more output
colcon build --packages-select my_robot_package --executor sequential
```

### Dependency Resolution

**Dependency Chain Visualization:**
```
┌─────────────────┐
│   my_robot_pkg  │ ←─ Top-level package
├─────────────────┤
│   Dependencies: │
│   - rclcpp      │
│   - sensor_msgs │
│   - my_msgs     │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   my_msgs       │ ←─ Custom message package
├─────────────────┤
│   Dependencies: │
│   - builtin_msgs│
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   builtin_msgs  │ ←─ Built-in ROS 2 messages
└─────────────────┘
```

## Exercises and Knowledge Checkpoints

### Knowledge Checkpoints

1. What is the purpose of the `package.xml` file?
2. What is the difference between build, exec, and test dependencies?
3. How do you create a new ROS 2 package using the command line?
4. What are the advantages of using `--symlink-install` during builds?
5. How do you specify build type and optimization flags?

### Hands-On Exercise

Create a complete ROS 2 package for a humanoid robot joint controller that includes:
1. A C++ node that publishes joint commands
2. A custom message definition for joint commands
3. A launch file to start the node
4. A configuration file with joint parameters
5. Unit tests for the node functionality
6. Proper package.xml and CMakeLists.txt files

Build the package and verify that all components work correctly.

## Summary

Building ROS 2 packages involves understanding the standard package structure, build system (ament), and best practices for organization and optimization. Proper package creation and building is fundamental for developing maintainable and reusable humanoid robot software components. The build system enables cross-compilation for target hardware and provides tools for testing and quality assurance.

## Next Steps

Continue to the next section: [URDF Robot Description](./urdf-robot-description.md)