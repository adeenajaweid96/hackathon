---
sidebar_position: 2.7
title: "ROS 2 Troubleshooting Guide"
---

# ROS 2 Troubleshooting Guide

## Overview

This chapter provides a comprehensive guide to troubleshooting common issues in ROS 2, with a focus on humanoid robotics applications. It covers diagnostic techniques, common problems, and systematic approaches to resolving issues that arise during development and deployment.

## Diagnostic Tools and Techniques

### Essential ROS 2 Tools

```
ROS 2 Diagnostic Ecosystem:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ros2 command  │    │   rqt tools     │    │   rviz2         │
│   line tools    │    │                 │    │                 │
│                 │    │  - rqt_graph    │    │  - Visualization │
│  - ros2 node    │    │  - rqt_plot     │    │  - TF debugging │
│  - ros2 topic   │    │  - rqt_console  │    │  - Message      │
│  - ros2 service │    │  - rqt_bag      │    │    inspection   │
│  - ros2 action  │    │  - rqt_reconfigure│   │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    System Monitoring                          │
│  - htop, iotop, nvidia-smi (for Jetson)                     │
│  - Network tools (netstat, iftop)                           │
│  - Process monitoring (ps, top)                             │
└─────────────────────────────────────────────────────────────────┘
```

### Node Diagnostics

**Checking Node Status:**
```bash
# List all active nodes
ros2 node list

# Get detailed information about a specific node
ros2 node info /my_robot_node

# Check node lifecycle state (for lifecycle nodes)
ros2 lifecycle list /my_lifecycle_node

# Echo node logs in real-time
ros2 topic echo /rosout
```

**Example diagnostic node:**
```cpp
#include <rclcpp/rclcpp.hpp>
#include <diagnostic_updater/diagnostic_updater.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>

class DiagnosticNode : public rclcpp::Node
{
public:
    DiagnosticNode() : Node("diagnostic_node")
    {
        // Initialize diagnostic updater
        updater_.setHardwareID("robot_hardware");

        // Add diagnostic checks
        updater_.add("System Health", this, &DiagnosticNode::check_system_health);
        updater_.add("Sensor Status", this, &DiagnosticNode::check_sensor_status);
        updater_.add("Communication", this, &DiagnosticNode::check_communication);

        // Timer for periodic diagnostics
        timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&DiagnosticNode::update_diagnostics, this));
    }

private:
    void check_system_health(diagnostic_updater::DiagnosticStatusWrapper &stat)
    {
        // Check system resources
        if (is_system_healthy()) {
            stat.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "System is healthy");
        } else {
            stat.summary(diagnostic_msgs::msg::DiagnosticStatus::ERROR, "System health issue detected");
        }

        // Add key-value pairs for detailed information
        stat.add("CPU Usage", get_cpu_usage());
        stat.add("Memory Usage", get_memory_usage());
        stat.add("Temperature", get_temperature());
    }

    void check_sensor_status(diagnostic_updater::DiagnosticStatusWrapper &stat)
    {
        // Check sensor availability and data quality
        if (sensors_operational()) {
            stat.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "All sensors operational");
        } else {
            stat.summary(diagnostic_msgs::msg::DiagnosticStatus::WARN, "Some sensors have issues");
        }
    }

    void check_communication(diagnostic_updater::DiagnosticStatusWrapper &stat)
    {
        // Check communication with other nodes
        if (communication_healthy()) {
            stat.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "Communication is healthy");
        } else {
            stat.summary(diagnostic_msgs::msg::DiagnosticStatus::ERROR, "Communication issue detected");
        }
    }

    void update_diagnostics()
    {
        updater_.update();
    }

    // Placeholder methods - implement based on your system
    bool is_system_healthy() { return true; }
    double get_cpu_usage() { return 25.0; }
    double get_memory_usage() { return 45.0; }
    double get_temperature() { return 42.0; }
    bool sensors_operational() { return true; }
    bool communication_healthy() { return true; }

    diagnostic_updater::Updater updater_;
    rclcpp::TimerBase::SharedPtr timer_;
};
```

## Common Communication Issues

### DDS Communication Problems

**Network Discovery Issues:**
```bash
# Check if nodes can discover each other
ros2 topic list
ros2 node list

# Check specific topic connections
ros2 topic info /topic_name

# Verify ROS domain ID consistency
echo $ROS_DOMAIN_ID

# Check network interfaces
ip addr show
```

**Firewall and Network Configuration:**
```bash
# For Fast DDS, check required ports
# Default port range: 1024-65535
# UDP multicast: 239.255.0.1:11811

# Temporarily disable firewall for testing (Ubuntu)
sudo ufw disable

# Or configure specific rules
sudo ufw allow from 192.168.1.0/24
sudo ufw allow 1024:65535/udp
sudo ufw allow 1024:65535/tcp
```

### Topic and Service Issues

**Message Synchronization Problems:**
```cpp
// Use message filters for synchronized processing
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

class SynchronizedProcessor : public rclcpp::Node
{
public:
    SynchronizedProcessor() : Node("sync_processor")
    {
        // Subscribe to multiple topics with synchronization
        image_sub_.subscribe(this, "camera/image_raw");
        info_sub_.subscribe(this, "camera/camera_info");

        // Synchronize with a small time tolerance (0.1 seconds)
        sync_ = std::make_shared<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo>>(
            image_sub_, info_sub_, 10);  // 10 messages max queue
        sync_->registerCallback(std::bind(&SynchronizedProcessor::callback, this,
                                        std::placeholders::_1, std::placeholders::_2));
    }

private:
    void callback(const sensor_msgs::msg::Image::SharedPtr& image,
                 const sensor_msgs::msg::CameraInfo::SharedPtr& info)
    {
        RCLCPP_INFO(this->get_logger(), "Received synchronized messages");
        // Process synchronized data
    }

    message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
    message_filters::Subscriber<sensor_msgs::msg::CameraInfo> info_sub_;
    std::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo>> sync_;
};
```

### Quality of Service (QoS) Mismatches

**QoS Configuration Issues:**
```cpp
// Common QoS configuration patterns
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>

class QoSConfiguredNode : public rclcpp::Node
{
public:
    QoSConfiguredNode() : Node("qos_configured_node")
    {
        // For sensor data (high frequency, best effort)
        auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(10))
            .best_effort()
            .durability_volatile();

        sensor_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "camera/image_raw", sensor_qos);

        // For critical commands (reliable, keep last)
        auto cmd_qos = rclcpp::QoS(rclcpp::KeepLast(1))
            .reliable()
            .durability_volatile();

        cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
            "cmd_vel", cmd_qos);

        // For map data (reliable, keep all)
        auto map_qos = rclcpp::QoS(rclcpp::KeepAll())
            .reliable()
            .transient_local();

        map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
            "map", map_qos);
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr sensor_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
};
```

## Performance Issues

### Memory and CPU Optimization

**Memory Profiling:**
```cpp
// Memory-efficient node implementation
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <memory>

class MemoryEfficientNode : public rclcpp::Node
{
public:
    MemoryEfficientNode() : Node("memory_efficient_node")
    {
        // Use small queue sizes to limit memory usage
        rclcpp::QoS qos_profile(1);  // Only keep 1 message in queue
        qos_profile.best_effort();  // Reduce memory for sensor data

        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "camera/image_raw", qos_profile,
            std::bind(&MemoryEfficientNode::image_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Memory-efficient node initialized");
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Process without copying data unnecessarily
        if (msg->width > 0 && msg->height > 0) {
            // Only process essential information
            last_image_info_.width = msg->width;
            last_image_info_.height = msg->height;
            last_image_info_.encoding = msg->encoding;

            // Avoid storing large data in class members
            // Process and publish results immediately
        }
    }

    struct ImageInfo {
        uint32_t width{0};
        uint32_t height{0};
        std::string encoding;
    } last_image_info_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};
```

**CPU Usage Optimization:**
```cpp
// CPU-optimized processing with threading
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <thread>
#include <mutex>
#include <queue>

class OptimizedProcessingNode : public rclcpp::Node
{
public:
    OptimizedProcessingNode() : Node("optimized_processing_node")
    {
        // Use callback groups for better threading control
        auto processing_callback_group = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        rclcpp::SubscriptionOptions options;
        options.callback_group = processing_callback_group;

        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "camera/image_raw", 10,
            std::bind(&OptimizedProcessingNode::image_callback, this, std::placeholders::_1),
            options);

        // Create separate thread for intensive processing
        processing_thread_ = std::thread(&OptimizedProcessingNode::process_images, this);

        RCLCPP_INFO(this->get_logger(), "Optimized processing node initialized");
    }

    ~OptimizedProcessingNode()
    {
        running_ = false;
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Quickly queue for processing, don't do heavy work in callback
        std::lock_guard<std::mutex> lock(queue_mutex_);
        image_queue_.push(msg);
    }

    void process_images()
    {
        while (running_) {
            sensor_msgs::msg::Image::SharedPtr msg;
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                if (!image_queue_.empty()) {
                    msg = image_queue_.front();
                    image_queue_.pop();
                }
            }

            if (msg) {
                // Perform intensive processing in separate thread
                perform_intensive_processing(msg);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Prevent busy waiting
        }
    }

    void perform_intensive_processing(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Heavy processing here - does not block the subscriber
        RCLCPP_DEBUG(this->get_logger(), "Processing image: %dx%d",
                   msg->width, msg->height);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    std::queue<sensor_msgs::msg::Image::SharedPtr> image_queue_;
    std::mutex queue_mutex_;
    std::thread processing_thread_;
    std::atomic<bool> running_{true};
};
```

## Hardware-Specific Issues

### Jetson Platform Troubleshooting

**Thermal and Power Management:**
```bash
# Check thermal status
sudo tegrastats

# Monitor power consumption
sudo tegrastats | grep -E "POM_5V_IN|VDD_IN"

# Check CPU/GPU frequencies
sudo tegrastats | grep -E "CPU@|GPU@"

# Apply jetson_clocks for consistent performance
sudo jetson_clocks

# Check power mode
sudo nvpmodel -q
```

**CUDA and GPU Issues:**
```bash
# Verify CUDA installation
nvidia-smi
nvcc --version

# Test CUDA samples
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery

# Check GPU memory usage
nvidia-smi -q -d MEMORY

# Verify ROS 2 CUDA support
printenv | grep CUDA
```

### Sensor-Specific Troubleshooting

**Camera Issues:**
```bash
# Check camera device availability
ls -l /dev/video*

# Test camera with v4l2
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext

# Check camera permissions
sudo usermod -a -G video $USER
```

**LiDAR Troubleshooting:**
```bash
# Check serial/USB connections
lsusb
dmesg | grep -i usb

# Check network LiDAR
ping <lidar_ip_address>
nmap -p <port> <lidar_ip_address>

# Verify baud rates and settings
stty -F /dev/ttyUSB0
```

## Build and Runtime Issues

### Compilation Problems

**Common CMake Issues:**
```cmake
# In CMakeLists.txt - proper dependency handling
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)  # For image processing

# Ensure dependencies are linked properly
ament_target_dependencies(your_node
  rclcpp
  sensor_msgs
  cv_bridge
)

# Handle OpenCV properly
find_package(OpenCV REQUIRED)
target_link_libraries(your_node ${OpenCV_LIBS})

# Install rules
install(TARGETS
  your_node
  DESTINATION lib/${PROJECT_NAME}
)
```

**Dependency Resolution:**
```bash
# Check package dependencies
rosdep check --from-paths src --ignore-src -r -y

# Install missing dependencies
rosdep install --from-paths src --ignore-src -r -y

# Verify package.xml dependencies
colcon build --packages-select your_package --cmake-args -DCMAKE_BUILD_TYPE=Debug
```

### Runtime Library Issues

**Library Path Problems:**
```bash
# Check library dependencies
ldd /path/to/your/binary

# Set library path if needed
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Check for missing libraries
ldconfig -p | grep <library_name>
```

## Launch File Issues

### Launch File Debugging

**Common Launch Issues:**
```python
# launch/debug_launch.py - Proper error handling in launch files
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, RegisterEventHandler
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from launch.events.process import ProcessExited

def generate_launch_description():
    # Declare launch arguments with defaults
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    log_level = LaunchConfiguration('log_level', default='info')

    # Create nodes with proper error handling
    perception_node = Node(
        package='my_robot_perception',
        executable='perception_node',
        name='perception_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'log_level': log_level}
        ],
        # Restart on failure
        respawn=True,
        respawn_delay=2,
        # Output configuration
        output='screen',
        # Additional configuration
        arguments=['--ros-args', '--log-level', log_level]
    )

    # Add event handlers for debugging
    def on_node_start(event, context):
        return [LogInfo(msg=['Perception node started with PID: ', event.pid])]

    def on_node_exit(event, context):
        return [LogInfo(msg=['Perception node exited with code: ', event.returncode])]

    # Register event handlers
    start_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=perception_node,
            on_start=on_node_start
        )
    )

    exit_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=perception_node,
            on_exit=on_node_exit
        )
    )

    return LaunchDescription([
        start_handler,
        exit_handler,
        perception_node
    ])
```

## Debugging Techniques

### Logging and Debugging

**Structured Logging:**
```cpp
#include <rclcpp/rclcpp.hpp>

class DebuggingNode : public rclcpp::Node
{
public:
    DebuggingNode() : Node("debugging_node")
    {
        // Set up different log levels
        this->get_logger();

        // Use different log levels appropriately
        RCLCPP_INFO(this->get_logger(), "Node initialized successfully");
        RCLCPP_DEBUG(this->get_logger(), "Detailed debug information: %d", some_value_);
        RCLCPP_WARN(this->get_logger(), "Warning about potential issue");
        RCLCPP_ERROR(this->get_logger(), "Error occurred: %s", error_message_.c_str());

        // Parameter-based debug level
        this->declare_parameter("debug_level", 0);
        debug_level_ = this->get_parameter("debug_level").as_int();
    }

private:
    int debug_level_{0};
    std::string error_message_;
    int some_value_{42};
};
```

**ROS 2 Logging Configuration:**
```bash
# Set log level for specific node
ROS_LOG_LEVEL=DEBUG ros2 run my_package my_node

# Or set for all nodes
export RCUTILS_LOGGING_SEVERITY_THRESHOLD=DEBUG

# Use ros2 run with log level
ros2 run my_package my_node --ros-args --log-level debug
```

### Remote Debugging

**GDB with ROS 2:**
```bash
# Attach debugger to running node
ros2 run --prefix 'gdb -ex run --args' my_package my_node

# Or use valgrind for memory issues
ros2 run --prefix 'valgrind --tool=memcheck' my_package my_node

# Use strace for system call debugging
strace -p $(pgrep -f my_node)
```

## Systematic Troubleshooting Approach

### Problem Diagnosis Framework

```
Systematic Troubleshooting Framework:
┌─────────────────┐
│   1. Identify   │ ←─ Gather symptoms and error messages
│   the Problem   │
└─────────┬───────┘
          ▼
┌─────────────────┐
│   2. Isolate    │ ←─ Determine scope and boundaries
│   the Scope     │
└─────────┬───────┘
          ▼
┌─────────────────┐
│   3. Formulate  │ ←─ Develop hypotheses about causes
│   Hypotheses    │
└─────────┬───────┘
          ▼
┌─────────────────┐
│   4. Test       │ ←─ Systematically test each hypothesis
│   Hypotheses    │
└─────────┬───────┘
          ▼
┌─────────────────┐
│   5. Implement  │ ←─ Apply the solution
│   Solution      │
└─────────┬───────┘
          ▼
┌─────────────────┐
│   6. Verify     │ ←─ Confirm the problem is resolved
│   Resolution    │
└─────────────────┘
```

### Common Troubleshooting Scenarios

**Scenario 1: Nodes Not Communicating**
```bash
# Step 1: Verify nodes are running
ros2 node list

# Step 2: Check if topics exist
ros2 topic list

# Step 3: Check topic connections
ros2 topic info /topic_name

# Step 4: Check ROS domain
echo $ROS_DOMAIN_ID

# Step 5: Verify network connectivity
ros2 topic echo /topic_name --field data
```

**Scenario 2: High CPU/Memory Usage**
```bash
# Monitor resource usage
htop
iotop

# Check specific ROS nodes
ros2 run topicos monitor

# Analyze node performance
ros2 run topic_tools delay /topic_name

# Check for memory leaks
valgrind --tool=memcheck ros2 run my_package my_node
```

**Scenario 3: Sensor Data Issues**
```bash
# Check sensor driver status
ros2 node info /sensor_driver_node

# Verify sensor data publication
ros2 topic echo /sensor_topic --field data

# Check sensor parameters
ros2 param list /sensor_driver_node

# Test sensor hardware directly
# (Use manufacturer's tools or direct interface)
```

## Prevention Strategies

### Best Practices for Avoiding Issues

**Code Quality:**
```cpp
// Proper error handling template
#include <rclcpp/rclcpp.hpp>

class RobustNode : public rclcpp::Node
{
public:
    RobustNode() : Node("robust_node")
    {
        try {
            initialize_components();
            setup_subscriptions();
            setup_publishers();
            setup_services();
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Initialization failed: %s", e.what());
            throw; // Re-throw to prevent node from starting in bad state
        }
    }

private:
    void initialize_components()
    {
        // Validate parameters before use
        this->declare_parameter("param_name", rclcpp::PARAMETER_DOUBLE);
        if (!this->has_parameter("param_name")) {
            throw std::runtime_error("Required parameter not found");
        }

        // Validate parameter values
        double param_value = this->get_parameter("param_name").as_double();
        if (param_value < 0.0 || param_value > 100.0) {
            throw std::runtime_error("Parameter value out of range");
        }
    }

    void setup_subscriptions()
    {
        try {
            subscription_ = this->create_subscription<std_msgs::msg::String>(
                "topic_name", 10,
                std::bind(&RobustNode::topic_callback, this, std::placeholders::_1));
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create subscription: %s", e.what());
            // Handle error appropriately
        }
    }

    void topic_callback(const std_msgs::msg::String::SharedPtr msg)
    {
        try {
            // Validate message content
            if (msg->data.empty()) {
                RCLCPP_WARN(this->get_logger(), "Received empty message");
                return;
            }

            // Process message
            process_message(msg);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error processing message: %s", e.what());
            // Continue operation, don't crash on single message error
        }
    }

    void process_message(const std_msgs::msg::String::SharedPtr msg)
    {
        // Actual message processing
    }

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};
```

**Configuration Management:**
```yaml
# config/robot_params.yaml
robust_node:
  ros__parameters:
    # Timeouts and limits
    connection_timeout: 5.0
    max_retries: 3
    retry_delay: 1.0

    # Resource limits
    max_queue_size: 10
    memory_limit_mb: 1024

    # Safety limits
    max_velocity: 1.0
    max_acceleration: 2.0
    safety_margin: 0.5
```

## Troubleshooting Checklist

### Pre-Deployment Checklist

```
Pre-Deployment Troubleshooting Checklist:
□ All nodes start without errors
□ Topics are properly connected
□ Parameters are validated and within ranges
□ QoS settings are compatible
□ Network connectivity is verified
□ Hardware drivers are loaded
□ Sensor data is being published
□ TF transforms are available
□ Safety systems are functional
□ Logging is configured appropriately
□ Resource usage is within limits
□ Error handling is implemented
```

### Runtime Monitoring Checklist

```
Runtime Monitoring Checklist:
□ Node status (ros2 node list)
□ Topic health (ros2 topic hz / ros2 topic info)
□ Memory usage (free, htop)
□ CPU usage (top, htop)
□ Network connectivity (ping, netstat)
□ Sensor data quality and rates
□ TF tree integrity (view_frames)
□ Log file analysis (ros2 topic echo /rosout)
□ Diagnostic topics (if available)
```

## Exercises and Knowledge Checkpoints

### Knowledge Checkpoints

1. What are the key diagnostic tools available in ROS 2?
2. How do you identify and resolve DDS communication issues?
3. What are the best practices for handling QoS mismatches?
4. How do you optimize resource usage in ROS 2 nodes?
5. What is the systematic approach to troubleshooting ROS 2 issues?

### Hands-On Exercise

Create a diagnostic system that:
1. Monitors multiple nodes and topics in a ROS 2 system
2. Detects common issues (missing connections, high resource usage, etc.)
3. Provides a unified diagnostic report
4. Implements automatic recovery for common failures
5. Logs diagnostic information for analysis

## Summary

Effective troubleshooting of ROS 2 systems requires a systematic approach, understanding of the middleware architecture, and familiarity with diagnostic tools. This chapter covered common issues in communication, performance, hardware integration, and deployment, along with strategies for prevention and resolution. Proper logging, monitoring, and error handling during development can prevent many runtime issues. The combination of ROS 2's built-in tools with system-level monitoring provides comprehensive diagnostic capabilities for humanoid robotics applications.

## Next Steps

Continue to the next section: [Humanoid Robotics Development](../humanoid-development/kinematics-dynamics.md)