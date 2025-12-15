---
sidebar_position: 2.5
title: "NVIDIA Jetson Deployment Guide"
---

# NVIDIA Jetson Deployment Guide

## Overview

This chapter provides a comprehensive guide for deploying ROS 2 applications on NVIDIA Jetson platforms, which are commonly used in humanoid robotics for their powerful AI and edge computing capabilities. The guide covers setup, optimization, and deployment strategies for Jetson Orin Nano and other Jetson devices.

## Jetson Platform Overview

### Jetson Hardware Specifications

```
NVIDIA Jetson Platform Comparison:
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│   Platform      │   Jetson Nano   │  Jetson Xavier  │  Jetson Orin    │
│                 │                 │     NX          │     Nano        │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ GPU             │ 128-core Maxwell│ 384-core Volta  │ 1024-core Ada   │
│                 │  @ 0.922 GHz    │  @ 1.38 GHz     │  @ 1.3 GHz      │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ CPU             │ Quad-core ARM   │ 8-core ARM      │ 8-core ARM      │
│                 │  A57 @ 1.43 GHz │  Carmel @ 2.26  │  Cortex-A78AE   │
│                 │                 │  GHz            │  @ 2.2 GHz      │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Memory          │ 4GB LPDDR4      │ 8GB LPDDR4x     │ 8GB LPDDR5      │
│                 │ @ 25.6 GB/s     │ @ 51.2 GB/s     │ @ 102.4 GB/s    │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ AI Performance  │ 0.5 TOPS        │ 22 TOPS         │ 77 TOPS         │
│                 │ (INT8)          │ (INT8)          │ (INT8)          │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

### Jetson in Humanoid Robotics Context

```
Humanoid Robot Computing Architecture:
┌─────────────────────────────────────────────────────────┐
│                    Cloud Services                         │
│  (Advanced AI, Data Processing, Remote Control)         │
└─────────────────┬─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                  Edge Computing                         │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │  Jetson Orin    │  │  Jetson Orin    │              │
│  │  (Navigation &  │  │  (Perception & │              │
│  │   Control)      │  │   Planning)     │              │
│  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                 Robot Hardware                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Motor      │  │  Sensor     │  │  Actuator   │    │
│  │  Control     │  │  Interface  │  │  Interface  │    │
│  │             │  │             │  │             │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## Setting Up Jetson for ROS 2

### Initial Setup and Configuration

**Step 1: Flash Jetson with JetPack SDK**
```bash
# Download and install JetPack SDK
# This includes:
# - Linux OS (Ubuntu-based)
# - CUDA toolkit
# - cuDNN
# - TensorRT
# - OpenCV
# - Multimedia API

# Flash the Jetson device using SDK Manager
jetpack_installer --flash-only
```

**Step 2: Install ROS 2**
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Set locale
sudo locale-gen en_US.UTF-8
export LANG=en_US.UTF-8

# Add ROS 2 apt repository
sudo apt update && sudo apt install curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 packages
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

# Initialize rosdep
sudo rosdep init
rosdep update

# Source ROS 2 environment
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Jetson-Specific Optimizations

**Power Mode Configuration:**
```bash
# Check current power mode
sudo nvpmodel -q

# Set to maximum performance mode (for compute-intensive tasks)
sudo nvpmodel -m 0  # 0 = MAXN mode, higher power consumption, maximum performance
# Alternative: sudo nvpmodel -m 1  # 1 = 15W mode, balanced performance

# Apply jetson_clocks for maximum performance during development/testing
sudo jetson_clocks
```

**Memory Management:**
```bash
# Configure swap space for memory-intensive operations
sudo fallocate -l 8G /mnt/swapfile
sudo chmod 600 /mnt/swapfile
sudo mkswap /mnt/swapfile
sudo swapon /mnt/swapfile

# Make swap permanent
echo '/mnt/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## ROS 2 Workspace Setup on Jetson

### Creating a Workspace

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build the workspace
colcon build --symlink-install

# Source the workspace
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Cross-Compilation Considerations

```
Development vs. Deployment Workflow:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Development   │    │   Cross-        │    │   Jetson        │
│   Machine       │───▶│   Compilation   │───▶│   Device        │
│   (x86_64)      │    │   (aarch64)     │    │   (aarch64)     │
│                 │    │                 │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │ Source    │  │    │  │ Build for │  │    │  │ Optimized │  │
│  │ Code      │──┼───▶│──│ aarch64   │──┼───▶│──│ Binaries  │  │
│  │ (.cpp,    │  │    │  │ Target    │  │    │  │           │  │
│  │ .py)      │  │    │  │           │  │    │  │           │  │
│  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Cross-compilation setup:**
```bash
# On development machine (x86_64)
# Install cross-compilation tools
sudo apt install g++-aarch64-linux-gnu

# Create toolchain file for CMake
cat > jetson_toolchain.cmake << EOF
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
EOF

# Build for Jetson target
colcon build --cmake-args -DCMAKE_TOOLCHAIN_FILE=jetson_toolchain.cmake
```

## Performance Optimization

### GPU Acceleration with CUDA

**CUDA Setup for ROS 2:**
```cpp
// Example: CUDA-accelerated image processing node
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

class GPUImageProcessor : public rclcpp::Node
{
public:
    GPUImageProcessor() : Node("gpu_image_processor")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "camera/image_raw", 10,
            std::bind(&GPUImageProcessor::image_callback, this, std::placeholders::_1));

        publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
            "camera/image_processed", 10);
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Convert ROS image to OpenCV
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // Process on GPU
        cv::cuda::GpuMat gpu_src, gpu_dst;
        gpu_src.upload(cv_ptr->image);

        // Example: GPU-accelerated Gaussian blur
        cv::cuda::bilateralFilter(gpu_src, gpu_dst, -1, 25, 25);

        // Download result
        cv::Mat result;
        gpu_dst.download(result);

        // Publish processed image
        cv_bridge::CvImage out_msg;
        out_msg.header = msg->header;
        out_msg.encoding = sensor_msgs::image_encodings::BGR8;
        out_msg.image = result;
        publisher_->publish(*out_msg.toImageMsg());
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};
```

### Memory Optimization

**Memory-efficient data handling:**
```python
# Python example: Efficient memory management in ROS 2 nodes
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
from collections import deque

class MemoryEfficientNode(Node):
    def __init__(self):
        super().__init__('memory_efficient_node')

        # Use memory-efficient data structures
        self.image_buffer = deque(maxlen=5)  # Fixed-size buffer

        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            1)  # Minimal queue size

        # Use callbacks that avoid unnecessary data copying
        self.processed_count = 0

    def image_callback(self, msg):
        # Process only when needed, not all the time
        if self.processed_count % 3 == 0:  # Process every 3rd image
            # Convert efficiently
            img_data = np.frombuffer(msg.data, dtype=np.uint8)
            img_data = img_data.reshape((msg.height, msg.width, 3))

            # Process and publish
            processed_img = self.process_image(img_data)

        self.processed_count += 1

    def process_image(self, image):
        # Implement efficient processing
        return cv2.GaussianBlur(image, (5, 5), 0)
```

### Real-Time Performance Configuration

**Real-time setup for Jetson:**
```bash
# Install real-time kernel patches (if available)
# Or configure kernel parameters for better real-time performance

# Increase process priority
echo "* soft rtprio 99" | sudo tee -a /etc/security/limits.conf
echo "* hard rtprio 99" | sudo tee -a /etc/security/limits.conf

# Configure CPU governor for performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable CPU idle states for critical applications
echo 0 | sudo tee /sys/devices/system/cpu/cpuidle/enable
```

## Deployment Strategies

### Containerization with Docker

**Dockerfile for Jetson deployment:**
```dockerfile
# Use NVIDIA's official ROS 2 container for Jetson
FROM nvcr.io/nvidia/ros:humble-ros-base-l4t-r35.2.1

# Set environment variables
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up ROS 2 workspace
WORKDIR /ros2_ws
COPY src /ros2_ws/src

# Build the workspace
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

# Source ROS 2 environment
ENV ROS_DISTRO=humble
ENV ROS_DOMAIN_ID=0

# Set entrypoint
ENTRYPOINT ["bash", "-c", "source /ros2_ws/install/setup.sh && exec \"$@\"", "--"]
```

**Docker Compose for multi-container deployment:**
```yaml
version: '3.8'
services:
  perception:
    build: .
    command: ros2 launch my_robot_perception perception.launch.py
    devices:
      - /dev/video0:/dev/video0  # Camera access
    environment:
      - DISPLAY=$DISPLAY
      - QT_X11_NO_MITSHM=1
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
    network_mode: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  navigation:
    build: .
    command: ros2 launch my_robot_navigation navigation.launch.py
    network_mode: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Launch File Optimization

**Optimized launch file for Jetson:**
```python
# jetson_optimized_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    log_level = LaunchConfiguration('log_level', default='info')

    # Set environment variables for Jetson optimization
    return LaunchDescription([
        # Set environment for optimized CUDA operations
        SetEnvironmentVariable(name='CUDA_CACHE_MAXSIZE', value='2147483648'),
        SetEnvironmentVariable(name='CUDA_CACHE_PATH', value='/tmp/.nv/'),

        # Perception node with GPU acceleration
        Node(
            package='my_robot_perception',
            executable='gpu_perception_node',
            name='gpu_perception_node',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'log_level': log_level},
                {'use_gpu': True},
                {'max_memory_mb': 2048}  # Limit memory usage
            ],
            # Configure resource limits
            respawn=True,
            respawn_delay=2,
            # Set CPU affinity if needed
            # prefix=['taskset -c 4-5'],  # Use specific CPU cores
            output='screen'
        ),

        # Navigation node
        Node(
            package='my_robot_navigation',
            executable='jetson_navigation_node',
            name='jetson_navigation_node',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'planner_frequency': 5.0},  # Lower frequency for Jetson
                {'controller_frequency': 20.0},
                {'use_costmaps': True}
            ],
            respawn=True,
            respawn_delay=2,
            output='screen'
        )
    ])
```

## Monitoring and Diagnostics

### System Monitoring

**Jetson-specific monitoring:**
```python
# jetson_monitor.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Temperature
from std_msgs.msg import Float32
import subprocess
import re

class JetsonMonitor(Node):
    def __init__(self):
        super().__init__('jetson_monitor')

        # Publishers for system metrics
        self.temp_pub = self.create_publisher(Temperature, 'system/temperature', 10)
        self.power_pub = self.create_publisher(Float32, 'system/power', 10)
        self.fan_pub = self.create_publisher(Float32, 'system/fan_speed', 10)

        # Timer for periodic monitoring
        self.timer = self.create_timer(1.0, self.monitor_system)

    def monitor_system(self):
        # Get GPU temperature
        try:
            result = subprocess.run(['cat', '/sys/devices/virtual/thermal/thermal_zone1/temp'],
                                  capture_output=True, text=True)
            temp = float(result.stdout.strip()) / 1000.0  # Convert from millidegrees

            temp_msg = Temperature()
            temp_msg.temperature = temp
            temp_msg.variance = 0.0
            self.temp_pub.publish(temp_msg)

            self.get_logger().info(f'GPU Temperature: {temp}°C')
        except Exception as e:
            self.get_logger().error(f'Error reading temperature: {e}')

        # Get power consumption (if available)
        try:
            result = subprocess.run(['sudo', 'tegrastats'],
                                  capture_output=True, text=True, timeout=1)
            # Parse tegrastats output for power consumption
            # This is a simplified example - actual parsing would be more complex
        except Exception as e:
            self.get_logger().error(f'Error reading power: {e}')

def main():
    rclpy.init()
    monitor = JetsonMonitor()
    rclpy.spin(monitor)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Performance Profiling

**Profiling ROS 2 nodes on Jetson:**
```bash
# Use ros2 topic hz to check message frequency
ros2 topic hz /camera/image_raw

# Use ros2 node info to check node status
ros2 node info /my_robot_node

# Use system tools for CPU/memory monitoring
htop
iotop
nvidia-smi  # For GPU monitoring

# Use ROS 2 tools for performance analysis
source /opt/ros/humble/setup.bash
ros2 run topic_tools relay /input_topic /output_topic --hz=10

# Custom profiling with ROS 2 lifecycle nodes
ros2 lifecycle list /my_lifecycle_node
```

## Troubleshooting Common Issues

### Performance Issues

**CPU Throttling Detection:**
```bash
# Check if CPU is throttling due to thermal issues
sudo tegrastats

# Look for "Thermal Throttling" or "CPU@X.XXXGHz" indicating throttling
# If throttling occurs, improve cooling or reduce computational load
```

**Memory Issues:**
```bash
# Monitor memory usage
free -h
cat /proc/meminfo

# If running out of memory, consider:
# 1. Reducing queue sizes in ROS 2 subscriptions
# 2. Using more efficient data structures
# 3. Adding swap space as shown earlier
```

### Common ROS 2 on Jetson Issues

**DDS Communication Issues:**
```bash
# If experiencing communication issues between nodes:
# 1. Check ROS_DOMAIN_ID consistency
echo $ROS_DOMAIN_ID

# 2. Verify network configuration
ip addr show

# 3. Use Fast DDS configuration optimized for Jetson
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

**CUDA Runtime Issues:**
```bash
# Verify CUDA installation
nvidia-smi
nvcc --version

# Check CUDA samples
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```

## Best Practices for Jetson Deployment

### Resource Management

```
Resource Allocation Strategy:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Critical      │    │   Standard      │    │   Background    │
│   Nodes         │    │   Nodes         │    │   Nodes         │
│   (High         │    │   (Medium       │    │   (Low          │
│   Priority)     │    │   Priority)     │    │   Priority)     │
│                 │    │                 │    │                 │
│ - Safety        │    │ - Perception    │    │ - Logging       │
│ - Control       │    │ - Planning      │    │ - Monitoring    │
│ - Communication │    │ - Navigation    │    │ - Housekeeping  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Priority-Based Scheduling                    │
└─────────────────────────────────────────────────────────────────┘
```

### Deployment Checklist

```
Jetson Deployment Checklist:
□ Hardware setup verified (power, cooling, connectivity)
□ JetPack SDK installed and configured
□ ROS 2 installed and tested
□ GPU acceleration enabled and tested
□ Power mode set appropriately
□ Thermal monitoring in place
□ Memory usage optimized
□ Network configuration verified
□ Launch files optimized for Jetson
□ Monitoring and diagnostics active
□ Backup and recovery procedures established
```

## Exercises and Knowledge Checkpoints

### Knowledge Checkpoints

1. What are the key differences between Jetson Nano, Xavier NX, and Orin Nano?
2. How do you configure power modes on Jetson devices for optimal performance?
3. What are the advantages of using containers for ROS 2 deployment on Jetson?
4. How do you optimize CUDA operations for robotic applications on Jetson?
5. What monitoring tools are available for Jetson-specific metrics?

### Hands-On Exercise

Deploy a simple ROS 2 package on a Jetson device that:
1. Subscribes to a camera topic
2. Performs GPU-accelerated image processing
3. Publishes the processed image
4. Includes system monitoring for temperature and power
5. Uses optimized launch files for Jetson hardware

## Summary

Deploying ROS 2 applications on NVIDIA Jetson platforms requires understanding of both ROS 2 concepts and Jetson-specific optimizations. Key considerations include GPU acceleration, memory management, thermal management, and power optimization. Containerization and proper resource management ensure reliable operation in humanoid robotics applications. The combination of ROS 2's flexibility with Jetson's AI capabilities enables sophisticated robotic systems at the edge.

## Next Steps

Continue to the next section: [Sensor Integration and Data Processing](./sensor-data-processing.md)