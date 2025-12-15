---
sidebar_position: 2.2
title: "Nodes, Topics & Services"
---

# Nodes, Topics & Services

## Overview

This chapter covers the fundamental building blocks of ROS 2: nodes, topics, and services. These components form the backbone of robot software architecture, enabling distributed processing and communication between different parts of a robotic system. Understanding these concepts is essential for developing humanoid robot applications that can effectively coordinate multiple subsystems.

## Nodes: The Building Blocks

### Node Definition and Purpose

A node is an executable process that works as part of a ROS 2 system. Nodes are the fundamental units of computation in ROS 2 and can:
- Publish data to topics
- Subscribe to topics to receive data
- Provide services
- Call services
- Execute actions

### Node Creation Patterns

**Lifecycle Node Pattern:**

```
┌─────────────────────────────────────────────────────────┐
│                    Node Creation                        │
│                                                         │
│  Start ──▶ Configure ──▶ Activate ──▶ Execute ──▶ Stop │
│    │           │             │           │         │   │
│    │           ▼             ▼           ▼         │   │
│    └───► Unconfigured ──► Inactive ──► Active ─────┘   │
│                           │             │              │
│                           │             ▼              │
│                           └───────► Deactivate ────────┘
│                                         │
│                                         ▼
│                                   Cleanup/Finalize
└─────────────────────────────────────────────────────────┘
```

### Node Implementation Examples

**C++ Node with Parameters:**

```cpp
#include "rclcpp/rclcpp.hpp"

class ParameterizedNode : public rclcpp::Node
{
public:
    ParameterizedNode() : Node("parameterized_node")
    {
        // Declare parameters with default values
        this->declare_parameter("frequency", 10.0);
        this->declare_parameter("topic_name", "default_topic");

        // Get parameter values
        double frequency = this->get_parameter("frequency").as_double();
        std::string topic_name = this->get_parameter("topic_name").as_string();

        // Create publisher with parameterized topic
        publisher_ = this->create_publisher<std_msgs::msg::String>(topic_name, 10);

        // Create timer with parameterized frequency
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / frequency)),
            std::bind(&ParameterizedNode::timer_callback, this));
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::String();
        message.data = "Parameterized message";
        publisher_->publish(message);
    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
};
```

**Python Node with Parameters:**

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ParameterizedNode(Node):
    def __init__(self):
        super().__init__('parameterized_node')

        # Declare parameters
        self.declare_parameter('frequency', 10.0)
        self.declare_parameter('topic_name', 'default_topic')

        # Get parameter values
        frequency = self.get_parameter('frequency').value
        topic_name = self.get_parameter('topic_name').value

        # Create publisher
        self.publisher = self.create_publisher(String, topic_name, 10)

        # Create timer
        timer_period = 1.0 / frequency
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Parameterized message'
        self.publisher.publish(msg)
```

## Topics: Publish-Subscribe Communication

### Topic Architecture

Topics enable asynchronous, many-to-many communication in ROS 2:

```
Multiple Publishers ──────┐
    │                    │
    ▼                    ▼
┌─────────────┐    ┌─────────────┐
│  Publisher  │    │  Publisher  │
│    A        │    │    B        │
│             │    │             │
│  /sensor_   │───▶│  /sensor_   │──┐
│  data       │    │  data       │  │
└─────────────┘    └─────────────┘  │
         │                   │       │
         └───────────────────┼───────┘
                             ▼
                       ┌─────────────┐
                       │   Topic     │
                       │  /sensor_   │
                       │  data       │
                       └─────────────┘
                             │
                             ▼
    ┌────────────────────────┼────────────────────────┐
    │                        │                        │
    ▼                        ▼                        ▼
┌─────────────┐        ┌─────────────┐        ┌─────────────┐
│ Subscriber  │        │ Subscriber  │        │ Subscriber  │
│    C        │        │    D        │        │    E        │
│             │        │             │        │             │
│ Receives    │        │ Receives    │        │ Receives    │
│ messages    │        │ messages    │        │ messages    │
└─────────────┘        └─────────────┘        └─────────────┘
```

### Topic QoS Configuration

**QoS Profile Selection Matrix:**

```
┌─────────────────────────────────────────────────────────┐
│  Data Type        │ Reliability │ Durability │ History │
├───────────────────┼─────────────┼─────────────┼─────────┤
│ Sensor Streams    │ Best Effort │ Volatile    │ Keep 1  │
│ Command Streams   │ Reliable    │ Volatile    │ Keep 1  │
│ Map/Configuration │ Reliable    │ Transient   │ Keep All│
│ Log/Debug         │ Best Effort │ Volatile    │ Keep 10 │
└───────────────────┴─────────────┴─────────────┴─────────┘
```

### Topic Implementation Examples

**C++ Publisher with Custom QoS:**

```cpp
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

class ImagePublisherNode : public rclcpp::Node
{
public:
    ImagePublisherNode() : Node("image_publisher")
    {
        // Create QoS profile for sensor data (high frequency, best effort)
        auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10))
            .best_effort()
            .durability_volatile();

        publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
            "camera/image_raw", qos_profile);

        // Timer to simulate image publishing
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33), // ~30 FPS
            std::bind(&ImagePublisherNode::publish_image, this));
    }

private:
    void publish_image()
    {
        auto msg = sensor_msgs::msg::Image();
        // Fill in image data (simplified)
        msg.header.stamp = this->now();
        msg.header.frame_id = "camera_frame";

        publisher_->publish(msg);
    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};
```

**Python Subscriber with Custom QoS:**

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class ImageSubscriberNode(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        # Create QoS profile matching publisher
        qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            qos_profile)

        self.subscription  # Prevent unused variable warning

    def image_callback(self, msg):
        self.get_logger().info(f'Received image: {msg.width}x{msg.height}')
```

## Services: Request-Response Communication

### Service Architecture

Services enable synchronous, request-response communication:

```
┌─────────────────┐    Request    ┌─────────────────┐
│   Service       │ ────────────▶ │   Service       │
│   Client        │               │   Server        │
│   Node          │               │   Node          │
│                 │               │                 │
│ Sends request   │               │ Receives        │
│ to service      │               │ request and     │
│ /get_map        │               │ processes it    │
└─────────────────┘               └─────────────────┘
         ▲                                │
         │                                │
         │ Response                       │ Process
         │                                │ request
         │                                ▼
┌─────────────────┐              ┌─────────────────┐
│   Service       │ ◀────────────│   Service       │
│   Client        │   Send       │   Server        │
│   Node          │   response   │   Node          │
│                 │              │                 │
│ Receives        │              │ Sends response  │
│ response        │              │ back to client  │
└─────────────────┘              └─────────────────┘
```

### Service Implementation Examples

**Service Definition (.srv file):**

```text
# GetMap.srv
# Request
string map_name
---
# Response
bool success
string error_message
sensor_msgs/PointCloud2 map_data
```

**C++ Service Server:**

```cpp
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/srv/get_map.hpp"

class MapServiceServer : public rclcpp::Node
{
public:
    MapServiceServer() : Node("map_service_server")
    {
        // Create service
        service_ = this->create_service<nav_msgs::srv::GetMap>(
            "get_map",
            std::bind(&MapServiceServer::handle_get_map,
                     this,
                     std::placeholders::_1,
                     std::placeholders::_2,
                     std::placeholders::_3));
    }

private:
    void handle_get_map(
        const std::shared_ptr<rmw_request_id_t> request_header,
        const std::shared_ptr<nav_msgs::srv::GetMap::Request> request,
        const std::shared_ptr<nav_msgs::srv::GetMap::Response> response)
    {
        RCLCPP_INFO(this->get_logger(), "Received map request: %s", request->map_name.c_str());

        // Process the request (simplified)
        response->map.header.frame_id = "map";
        response->map.info.resolution = 0.05;
        response->map.info.width = 100;
        response->map.info.height = 100;
        response->success = true;
        response->status.message = "Map retrieved successfully";
    }

    rclcpp::Service<nav_msgs::srv::GetMap>::SharedPtr service_;
};
```

**Python Service Client:**

```python
import rclpy
from rclpy.node import Node
from nav_msgs.srv import GetMap

class MapServiceClient(Node):
    def __init__(self):
        super().__init__('map_service_client')
        self.cli = self.create_client(GetMap, 'get_map')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.req = GetMap.Request()

    def send_request(self, map_name):
        self.req.map_name = map_name
        self.future = self.cli.call_async(self.req)
        return self.future

def main():
    rclpy.init()

    client = MapServiceClient()
    future = client.send_request('main_map')

    rclpy.spin_until_future_complete(client, future)

    if future.result() is not None:
        response = future.result()
        if response.success:
            print(f'Map retrieved successfully: {response.map.info.width}x{response.map.info.height}')
        else:
            print(f'Failed to get map: {response.status.message}')
    else:
        print('Service call failed')

    client.destroy_node()
    rclpy.shutdown()
```

## Best Practices

### Node Design Principles

**Modular Node Design:**

```
┌─────────────────────────────────────────────────────────┐
│                Recommended Node Design                  │
│                                                         │
│  Single Responsibility: Each node should have one      │
│  clear purpose (e.g., sensor driver, controller,       │
│  planner)                                              │
│                                                         │
│  Loose Coupling: Nodes should communicate through       │
│  well-defined interfaces (topics, services, actions)   │
│                                                         │
│  High Cohesion: All functionality within a node         │
│  should be closely related                             │
└─────────────────────────────────────────────────────────┘
```

### Topic Design Guidelines

1. **Use descriptive topic names**: Follow naming conventions like `/sensor_name/data_type`
2. **Choose appropriate QoS settings**: Match communication requirements
3. **Consider message frequency**: Avoid overwhelming the system
4. **Use standard message types**: When possible, use built-in ROS 2 message types

### Service Design Guidelines

1. **Use for synchronous operations**: When you need guaranteed response
2. **Keep requests/responses small**: Avoid large data transfers
3. **Handle errors gracefully**: Return appropriate error codes
4. **Consider timeout handling**: Implement proper timeout mechanisms

## Common Patterns in Humanoid Robotics

### Sensor Data Aggregation

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  IMU Driver     │    │ Camera Driver   │    │ LiDAR Driver    │
│  Node           │    │  Node           │    │  Node           │
│  /imu/data      │    │  /camera/image  │    │  /lidar/points │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   Sensor Aggregator     │
                    │   Node                  │
                    │   /sensors/fused_data   │
                    └─────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Perception System     │
                    │   Node                  │
                    └─────────────────────────┘
```

### Command Distribution

```
┌─────────────────┐
│  High-Level     │
│  Planner Node   │
│  /cmd/move_base │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐    ┌─────────────────┐
│  Command        │    │  Command        │
│  Router Node    │    │  Router Node    │
│  /cmd/body      │    │  /cmd/arms      │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Body Control   │    │  Left Arm       │    │  Right Arm      │
│  Node           │    │  Control Node   │    │  Control Node   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Exercises and Knowledge Checkpoints

### Knowledge Checkpoints

1. What is the difference between a publisher and a subscriber?
2. When would you use a service instead of a topic?
3. What are the key QoS settings and when would you use each?
4. What is the purpose of the node lifecycle?
5. How do parameters differ from topics in ROS 2?

### Hands-On Exercise

Create a simple robot controller system with:
1. A sensor node that publishes random sensor data
2. A processing node that subscribes to sensor data and processes it
3. A service that allows external nodes to request the latest processed data
4. A client node that calls the service and displays the result

## Summary

Nodes, topics, and services form the core communication infrastructure of ROS 2. Understanding how to properly design and implement these components is crucial for creating robust humanoid robot applications. The publish-subscribe pattern enables asynchronous data sharing, while services provide synchronous request-response communication for critical operations.

## Next Steps

Continue to the next section: [Building ROS 2 Packages](./building-ros-packages.md)