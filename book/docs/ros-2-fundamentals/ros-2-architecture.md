---
sidebar_position: 2.1
title: "ROS 2 Architecture & Concepts"
---

# ROS 2 Architecture & Concepts

## Overview

Robot Operating System 2 (ROS 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms. This chapter introduces the fundamental concepts and architecture of ROS 2, specifically focusing on applications for humanoid robotics.

## What is ROS 2?

### Definition and Purpose

ROS 2 is a middleware framework that provides services designed for a heterogeneous computer cluster, including:
- Hardware abstraction
- Device drivers
- Libraries for implementing commonly used functionality
- Message-passing between processes
- Package management

### Key Improvements Over ROS 1

**ROS 2 vs. ROS 1:**

```
ROS 1 Architecture:                    ROS 2 Architecture:
┌─────────────────┐                    ┌─────────────────┐
│   Master-based  │                    │  DDS-based      │
│   (Centralized) │                    │  (Distributed)  │
│                 │                    │                 │
│  Single point   │                    │  Fault tolerant │
│  of failure     │                    │  communication  │
└─────────────────┘                    └─────────────────┘
         │                                         │
         ▼                                         ▼
┌─────────────────┐                    ┌─────────────────┐
│  Single-threaded│                    │  Multi-threaded │
│  communication  │                    │  communication  │
│                 │                    │                 │
│  Limited real-  │                    │  Real-time      │
│  time support   │                    │  capable        │
└─────────────────┘                    └─────────────────┘
```

## Core Architecture Concepts

### Nodes

A node is an executable that uses ROS 2 to communicate with other nodes. In humanoid robotics, nodes typically represent:
- Individual sensors (cameras, LiDAR, IMU)
- Processing units (perception, planning, control)
- Hardware interfaces (motor controllers, grippers)

**Node Lifecycle:**

```
    Unconfigured
         │
         ▼
    ┌─────────────┐
    │  configure  │
    └─────────────┘
         │
         ▼
    Inactive ──────┐
         │         │
         │         ▼
    ┌─────────────┐ │
    │   activate  │ │
    └─────────────┘ │
         │         │
         ▼         │
      Active ◄─────┤
         │         │
         │         │
    ┌─────────────┤ │
    │ deactivate  │◄┘
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │   cleanup   │
    └─────────────┘
         │
         ▼
    Finalized
```

### Packages

A package is the fundamental unit of organization in ROS 2. A package typically contains:
- Source code (C++/Python)
- Launch files
- Configuration files
- Message/Service/Action definitions
- Documentation

**Package Structure:**
```
ros2_package/
├── CMakeLists.txt          # Build configuration for C++
├── package.xml            # Package metadata
├── src/                   # Source code
│   ├── node1.cpp
│   └── node2.py
├── include/               # Header files
├── launch/                # Launch files
│   └── my_launch.py
├── config/                # Configuration files
│   └── params.yaml
├── msg/                   # Custom message definitions
│   └── MyMessage.msg
└── test/                  # Unit tests
```

### Topics and Messages

Topics are named buses over which nodes exchange messages. Messages are the data structures that are passed between nodes.

**Topic Communication Pattern:**

```
Publisher Node A     Publisher Node B
       │                     │
       │ publishes           │ publishes
       │ to /sensor_data     │ to /cmd_vel
       │                     │
       └─────────┬───────────┘
                 │
    ┌─────────────────────────┐
    │     DDS Middleware      │
    │   (Data Distribution    │
    │    Service)             │
    └─────────┬───────────────┘
              │
Subscriber Node C        Subscriber Node D
  - Subscribes to        - Subscribes to
    /sensor_data           /cmd_vel
  - Processes data       - Processes commands
  - May republish        - Sends to hardware
```

## Communication Patterns

### Publish-Subscribe Pattern

The most common communication pattern in ROS 2:

```
┌─────────────────┐
│   Publisher     │
│   Node          │
│                 │
│ Publishes data  │
│ to topic        │
│ /sensor_data    │
└─────────┬───────┘
          │
          │ (Message)
          │
          ▼
    ┌─────────────┐
    │   Topic     │
    │   /sensor_data│
    │             │
    └─────────────┘
          │
          │ (Message)
          │
    ┌─────▼─────────┐    ┌──────────────┐
    │ Subscriber    │    │ Subscriber    │
    │ Node 1        │    │ Node 2        │
    │               │    │               │
    │ Receives data │    │ Receives data │
    │ from topic    │    │ from topic    │
    └───────────────┘    └───────────────┘
```

### Service-Client Pattern

For request-response communication:

```
┌─────────────────┐              ┌─────────────────┐
│   Client        │              │   Service       │
│   Node          │              │   Server        │
│                 │              │                 │
│ Sends request   │─────────────▶│ Processes       │
│ to service      │              │ request         │
│ /get_map        │              │ /get_map        │
└─────────────────┘              └─────────────────┘
                                        │
                                        │
                                        ▼
                              ┌─────────────────┐
                              │ Returns         │
                              │ response        │
                              └─────────────────┘
                                        │
                                        │
                                        ▼
                              ┌─────────────────┐
                              │   Client        │
                              │   Node          │
                              │                 │
                              │ Receives        │
                              │ response        │
                              └─────────────────┘
```

### Action-Based Communication

For long-running tasks with feedback:

```
┌─────────────────┐              ┌─────────────────┐
│   Action        │              │   Action        │
│   Client        │              │   Server        │
│                 │              │                 │
│ Sends goal      │─────────────▶│ Receives goal   │
│ to action       │              │ and starts      │
│ /move_to_pose   │              │ execution       │
└─────────────────┘              └─────────────────┘
                                        │
                                        │
                                        ▼
                              ┌─────────────────┐
                              │ Executes action │
                              │ and sends       │
                              │ feedback        │
                              └─────────┬───────┘
                                        │
                                        │ (Feedback)
                                        ▼
                              ┌─────────────────┐
                              │   Action        │
                              │   Client        │
                              │                 │
                              │ Receives        │
                              │ feedback        │
                              └─────────────────┘
                                        │
                                        │
                                        ▼
                              ┌─────────────────┐
                              │   Action        │
                              │   Server        │
                              │                 │
                              │ Sends result    │
                              │ when complete   │
                              └─────────────────┘
                                        │
                                        │ (Result)
                                        ▼
                              ┌─────────────────┐
                              │   Action        │
                              │   Client        │
                              │                 │
                              │ Receives        │
                              │ result          │
                              └─────────────────┘
```

## Quality of Service (QoS) Settings

QoS settings allow you to control the behavior of communication between nodes:

**QoS Settings Overview:**

```
QoS Configuration Matrix:
┌─────────────────────────────────────────────────────────┐
│  Data Type      │ Reliability │ Durability │ History   │
├─────────────────┼─────────────┼─────────────┼───────────┤
│ Sensor Data     │ Best Effort │ Volatile    │ Keep Last │
│ Critical Cmds   │ Reliable    │ Volatile    │ Keep Last │
│ Map Data        │ Reliable    │ Transient   │ Keep All  │
│ Log Data        │ Best Effort │ Volatile    │ Keep Last │
└─────────────────┴─────────────┴─────────────┴───────────┘
```

### Reliability Policy

- **Reliable**: All messages are guaranteed to be delivered
- **Best Effort**: Messages may be lost, but faster delivery

### Durability Policy

- **Volatile**: Only new messages after subscription
- **Transient Local**: All messages including past ones

### History Policy

- **Keep Last**: Only the most recent messages
- **Keep All**: All messages are stored

## DDS (Data Distribution Service)

### DDS Role in ROS 2

DDS is the middleware that ROS 2 uses for communication:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ROS 2 Node    │    │    DDS          │    │   ROS 2 Node    │
│   A             │    │   Implementation│    │   B             │
│                 │    │   (Fast DDS,    │    │                 │
│  ROS 2 API      │───▶│   Cyclone DDS,  │◀───│  ROS 2 API      │
│  calls          │    │   etc.)         │    │  calls          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Application     │    │ Data            │    │ Application     │
│ Data            │    │ Exchange        │    │ Data            │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Practical Example: Creating a Simple Node

### C++ Node Structure

```cpp
#include "rclcpp/rclcpp.hpp"

class SimpleNode : public rclcpp::Node
{
public:
    SimpleNode() : Node("simple_node")
    {
        // Create a publisher
        publisher_ = this->create_publisher<std_msgs::msg::String>(
            "topic_name", 10);

        // Create a timer
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500),
            std::bind(&SimpleNode::timer_callback, this));
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::String();
        message.data = "Hello World";
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);
    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SimpleNode>());
    rclcpp::shutdown();
    return 0;
}
```

### Python Node Structure

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SimpleNode(Node):
    def __init__(self):
        super().__init__('simple_node')
        self.publisher_ = self.create_publisher(String, 'topic_name', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World'
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    simple_node = SimpleNode()
    rclpy.spin(simple_node)
    simple_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercises and Knowledge Checkpoints

### Knowledge Checkpoints

1. What is the main difference between ROS 1 and ROS 2 architecture?
2. What are the three main communication patterns in ROS 2?
3. When would you use Reliable vs. Best Effort QoS settings?
4. What is the purpose of the DDS middleware in ROS 2?
5. What are the different states in a ROS 2 node lifecycle?

### Hands-On Exercise

Create a simple ROS 2 publisher-subscriber pair that:
1. Publisher node publishes a counter value every second
2. Subscriber node receives and logs the counter value
3. Use appropriate QoS settings for the communication

## Summary

ROS 2 provides a robust middleware framework for robot software development. Understanding its architecture, communication patterns, and QoS settings is crucial for developing reliable humanoid robot applications. The distributed nature of ROS 2, based on DDS, provides better fault tolerance and real-time capabilities compared to ROS 1.

## Next Steps

Continue to the next section: [Nodes, Topics, Services](./nodes-topics-services.md)