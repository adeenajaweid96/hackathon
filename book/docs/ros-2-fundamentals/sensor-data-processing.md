---
sidebar_position: 2.6
title: "Sensor Data Processing"
---

# Sensor Data Processing

## Overview

This chapter covers the processing of sensor data in ROS 2, which is fundamental for humanoid robots to perceive and interact with their environment. We'll explore how to handle various sensor types, process data efficiently, and integrate multiple sensors for comprehensive environmental awareness.

## Sensor Types in Humanoid Robotics

### Common Robot Sensors

```
Humanoid Robot Sensor Suite:
┌─────────────────────────────────────────────────────────┐
│                    Vision Sensors                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  RGB        │  │  Depth      │  │  Thermal    │     │
│  │  Camera     │  │  Camera     │  │  Camera     │     │
│  │             │  │             │  │             │     │
│  │  640x480    │  │  320x240    │  │  160x120    │     │
│  │  30 FPS     │  │  15 FPS     │  │  9 FPS      │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│                   Distance Sensors                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  LiDAR      │  │  Ultrasonic │  │  Infrared   │     │
│  │  (2D/3D)    │  │  Rangefinder│  │  Distance   │     │
│  │             │  │             │  │  Sensor     │     │
│  │  360° FOV   │  │  10m range  │  │  1m range   │     │
│  │  10-20Hz    │  │  10-50Hz    │  │  100Hz      │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│                   Inertial Sensors                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  IMU         │  │  Gyroscope  │  │  Acceler-   │     │
│  │  (6-axis)    │  │             │  │  ometer     │     │
│  │             │  │             │  │             │     │
│  │  Accel +    │  │  Angular    │  │  Linear     │     │
│  │  Gyro       │  │  Velocity   │  │  Accel      │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## Sensor Data Handling Patterns

### Sensor Message Types

ROS 2 provides standard message types for common sensors:

```cpp
// Common sensor message types in ROS 2
#include "sensor_msgs/msg/image.hpp"           // Camera images
#include "sensor_msgs/msg/point_cloud2.hpp"    // 3D point clouds
#include "sensor_msgs/msg/laser_scan.hpp"      // LiDAR/Laser data
#include "sensor_msgs/msg/imu.hpp"             // Inertial measurement
#include "sensor_msgs/msg/joint_state.hpp"     // Joint positions
#include "sensor_msgs/msg/battery_state.hpp"   // Battery information
```

### Basic Sensor Data Subscriber

```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>

class SensorProcessor : public rclcpp::Node
{
public:
    SensorProcessor() : Node("sensor_processor")
    {
        // Subscribe to laser scan data
        laser_subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "scan", 10,
            std::bind(&SensorProcessor::laser_callback, this, std::placeholders::_1));

        // Subscribe to image data
        image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "camera/image_raw", 10,
            std::bind(&SensorProcessor::image_callback, this, std::placeholders::_1));

        // Subscribe to IMU data
        imu_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "imu/data", 10,
            std::bind(&SensorProcessor::imu_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Sensor processor initialized");
    }

private:
    void laser_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(),
            "Received laser scan: %zu points, range: %.2f to %.2f m",
            msg->ranges.size(), msg->range_min, msg->range_max);

        // Process laser data
        process_obstacle_detection(msg);
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(),
            "Received image: %dx%d, encoding: %s",
            msg->width, msg->height, msg->encoding.c_str());

        // Process image data
        process_image_data(msg);
    }

    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(),
            "Received IMU data: angular velocity (%.3f, %.3f, %.3f)",
            msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

        // Process IMU data
        process_imu_data(msg);
    }

    void process_obstacle_detection(const sensor_msgs::msg::LaserScan::SharedPtr scan)
    {
        // Simple obstacle detection logic
        for (size_t i = 0; i < scan->ranges.size(); ++i) {
            if (scan->ranges[i] < 1.0 && scan->ranges[i] > scan->range_min) {
                RCLCPP_WARN(this->get_logger(), "Obstacle detected at angle: %.2f",
                           scan->angle_min + i * scan->angle_increment);
                break;
            }
        }
    }

    void process_image_data(const sensor_msgs::msg::Image::SharedPtr image)
    {
        // Image processing would go here
        // For now, just log basic info
    }

    void process_imu_data(const sensor_msgs::msg::Imu::SharedPtr imu)
    {
        // IMU processing would go here
        // Calculate orientation, detect falls, etc.
    }

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
};
```

## Image Processing Pipeline

### Camera Data Processing

```
Image Processing Pipeline:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Image     │───▶│   Preprocess    │───▶│   Feature       │
│   Acquisition   │    │   & Filter      │    │   Extraction    │
│                 │    │                 │    │                 │
│  - Synchronization│   - Noise reduction│    - Edge detection │
│  - Calibration  │    - Enhancement    │    - Corner detect  │
│  - Rectification│    - Normalization  │    - Blob analysis  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Object        │───▶│   Recognition   │───▶│   Decision      │
│   Detection     │    │   & Tracking    │    │   Making        │
│                 │    │                 │    │                 │
│  - CNN-based    │    - KLT tracking   │    - Action         │
│  - YOLO, SSD    │    - Kalman filter  │    - Control output │
│  - Semantic seg │    - Particle filter│    - Behavior tree  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Image Processing with OpenCV

```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class ImageProcessor : public rclcpp::Node
{
public:
    ImageProcessor() : Node("image_processor")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "camera/image_raw", 10,
            std::bind(&ImageProcessor::image_callback, this, std::placeholders::_1));

        publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
            "camera/image_processed", 10);

        RCLCPP_INFO(this->get_logger(), "Image processor initialized");
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            // Convert ROS image to OpenCV
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

            // Apply image processing
            cv::Mat processed_image = process_image(cv_ptr->image);

            // Convert back to ROS image
            cv_bridge::CvImage out_msg;
            out_msg.header = msg->header;
            out_msg.encoding = sensor_msgs::image_encodings::BGR8;
            out_msg.image = processed_image;

            // Publish processed image
            publisher_->publish(*out_msg.toImageMsg());
        }
        catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    cv::Mat process_image(const cv::Mat& input)
    {
        cv::Mat output;

        // Apply Gaussian blur to reduce noise
        cv::GaussianBlur(input, output, cv::Size(5, 5), 0);

        // Convert to HSV for color-based processing
        cv::Mat hsv;
        cv::cvtColor(output, hsv, cv::COLOR_BGR2HSV);

        // Define color range for object detection (e.g., red object)
        cv::Scalar lower_red1(0, 50, 50);
        cv::Scalar upper_red1(10, 255, 255);
        cv::Scalar lower_red2(170, 50, 50);
        cv::Scalar upper_red2(180, 255, 255);

        cv::Mat mask1, mask2, mask;
        cv::inRange(hsv, lower_red1, upper_red1, mask1);
        cv::inRange(hsv, lower_red2, upper_red2, mask2);
        mask = mask1 | mask2;

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        // Draw contours on output image
        cv::drawContours(output, contours, -1, cv::Scalar(0, 255, 0), 2);

        return output;
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};
```

## LiDAR Processing

### Laser Scan Data Processing

```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class LaserProcessor : public rclcpp::Node
{
public:
    LaserProcessor() : Node("laser_processor")
    {
        scan_subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "scan", 10,
            std::bind(&LaserProcessor::scan_callback, this, std::placeholders::_1));

        cloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "scan_cloud", 10);

        RCLCPP_INFO(this->get_logger(), "Laser processor initialized");
    }

private:
    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr scan)
    {
        // Convert laser scan to point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

        for (size_t i = 0; i < scan->ranges.size(); ++i) {
            float range = scan->ranges[i];
            if (range >= scan->range_min && range <= scan->range_max) {
                float angle = scan->angle_min + i * scan->angle_increment;

                pcl::PointXYZ point;
                point.x = range * cos(angle);
                point.y = range * sin(angle);
                point.z = 0.0;

                cloud->points.push_back(point);
            }
        }

        cloud->width = cloud->points.size();
        cloud->height = 1;
        cloud->is_dense = false;

        // Convert to ROS message
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud, cloud_msg);
        cloud_msg.header = scan->header;

        cloud_publisher_->publish(cloud_msg);

        // Perform obstacle detection
        detect_obstacles(scan);
    }

    void detect_obstacles(const sensor_msgs::msg::LaserScan::SharedPtr scan)
    {
        // Simple obstacle detection in front of robot
        size_t front_index = scan->ranges.size() / 2;
        float front_distance = scan->ranges[front_index];

        if (front_distance < 1.0 && front_distance > scan->range_min) {
            RCLCPP_WARN(this->get_logger(), "Obstacle detected in front: %.2f m", front_distance);
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_publisher_;
};
```

## Sensor Fusion

### Multi-Sensor Integration

```
Sensor Fusion Architecture:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vision        │    │   LiDAR         │    │   IMU           │
│   (Camera)      │    │   (Range)       │    │   (Inertial)    │
│                 │    │                 │    │                 │
│  - Object       │    │  - Obstacle     │    │  - Orientation │
│    Detection    │    │    Detection    │    │  - Motion      │
│  - Semantic     │    │  - Mapping      │    │  - Calibration │
│    Segmentation │    │  - Localization │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   Sensor Fusion         │
                    │   Algorithm             │
                    │   (Kalman Filter,      │
                    │    Particle Filter,    │
                    │    Extended Kalman     │
                    │    Filter, etc.)       │
                    └─────────┬───────────────┘
                              ▼
                    ┌─────────────────────────┐
                    │   Fused State           │
                    │   Estimate              │
                    │   - Position            │
                    │   - Velocity            │
                    │   - Orientation         │
                    │   - Confidence          │
                    └─────────────────────────┘
```

### Basic Sensor Fusion Node

```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

class SensorFusion : public rclcpp::Node
{
public:
    SensorFusion() : Node("sensor_fusion")
    {
        // Initialize transform broadcaster
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        // Subscribe to sensor data
        imu_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "imu/data", 10,
            std::bind(&SensorFusion::imu_callback, this, std::placeholders::_1));

        scan_subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "scan", 10,
            std::bind(&SensorFusion::scan_callback, this, std::placeholders::_1));

        // Publisher for fused pose
        pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "fused_pose", 10);

        // Initialize pose estimate
        current_pose_.pose.position.x = 0.0;
        current_pose_.pose.position.y = 0.0;
        current_pose_.pose.position.z = 0.0;
        current_pose_.pose.orientation.w = 1.0;
        current_pose_.pose.orientation.x = 0.0;
        current_pose_.pose.orientation.y = 0.0;
        current_pose_.pose.orientation.z = 0.0;

        RCLCPP_INFO(this->get_logger(), "Sensor fusion node initialized");
    }

private:
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        // Update orientation from IMU
        current_pose_.pose.orientation = msg->orientation;

        // Integrate angular velocity for position estimation (simplified)
        // In practice, you'd use a proper filter like EKF
        auto now = this->now();
        double dt = (now - last_update_time_).seconds();

        if (dt > 0) {
            // Simple integration of angular velocity (not accurate for real use)
            current_pose_.header.stamp = now;
            current_pose_.header.frame_id = "odom";

            // Publish updated pose
            pose_publisher_->publish(current_pose_);

            // Broadcast transform
            publish_transform();
        }

        last_update_time_ = now;
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        // Use laser data for position correction
        // This is a simplified example - real fusion would use complex algorithms

        // For now, just log that we received scan data
        RCLCPP_DEBUG(this->get_logger(), "Received laser scan for fusion");
    }

    void publish_transform()
    {
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = this->now();
        transform.header.frame_id = "odom";
        transform.child_frame_id = "base_link";

        transform.transform.translation.x = current_pose_.pose.position.x;
        transform.transform.translation.y = current_pose_.pose.position.y;
        transform.transform.translation.z = current_pose_.pose.position.z;

        transform.transform.rotation = current_pose_.pose.orientation;

        tf_broadcaster_->sendTransform(transform);
    }

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscription_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    geometry_msgs::msg::PoseStamped current_pose_;
    rclcl::Time last_update_time_;
};
```

## Real-Time Processing Considerations

### Efficient Data Processing

```
Real-Time Processing Pipeline:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Input    │───▶│   Processing    │───▶│   Output        │
│   (Sensors)     │    │   (Algorithms)  │    │   (Actuators)   │
│                 │    │                 │    │                 │
│  - Rate control │    - Threading      │    - Priority       │
│  - Buffering    │    - Optimization   │    - Scheduling     │
│  - Filtering    │    - Parallelism    │    - Timing         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Real-Time Constraints                        │
│  - Deterministic timing                                       │
│  - Low latency processing                                     │
│  - Predictable memory usage                                   │
│  - Minimal jitter                                             │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-threaded Sensor Processing

```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <thread>
#include <mutex>
#include <queue>

class MultiThreadSensorProcessor : public rclcpp::Node
{
public:
    MultiThreadSensorProcessor() : Node("multi_thread_sensor_processor")
    {
        // Create separate callback groups for different sensors
        auto sensor_callback_group = this->create_callback_group(
            rclcpp::CallbackGroupType::Reentrant);

        // Separate executors for different sensor types
        rclcpp::SubscriptionOptions sensor_options;
        sensor_options.callback_group = sensor_callback_group;

        // Subscribe to different sensors
        image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "camera/image_raw", 10,
            std::bind(&MultiThreadSensorProcessor::image_callback, this, std::placeholders::_1),
            sensor_options);

        scan_subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "scan", 10,
            std::bind(&MultiThreadSensorProcessor::scan_callback, this, std::placeholders::_1),
            sensor_options);

        // Start processing threads
        image_processing_thread_ = std::thread(&MultiThreadSensorProcessor::process_images, this);
        scan_processing_thread_ = std::thread(&MultiThreadSensorProcessor::process_scans, this);

        RCLCPP_INFO(this->get_logger(), "Multi-threaded sensor processor initialized");
    }

    ~MultiThreadSensorProcessor()
    {
        running_ = false;
        if (image_processing_thread_.joinable()) {
            image_processing_thread_.join();
        }
        if (scan_processing_thread_.joinable()) {
            scan_processing_thread_.join();
        }
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(image_queue_mutex_);
        image_queue_.push(msg);
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(scan_queue_mutex_);
        scan_queue_.push(msg);
    }

    void process_images()
    {
        while (running_) {
            sensor_msgs::msg::Image::SharedPtr msg;
            {
                std::lock_guard<std::mutex> lock(image_queue_mutex_);
                if (!image_queue_.empty()) {
                    msg = image_queue_.front();
                    image_queue_.pop();
                }
            }

            if (msg) {
                // Process image (in real application, this would be computationally intensive)
                RCLCPP_DEBUG(this->get_logger(), "Processing image: %dx%d",
                           msg->width, msg->height);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Prevent busy waiting
        }
    }

    void process_scans()
    {
        while (running_) {
            sensor_msgs::msg::LaserScan::SharedPtr msg;
            {
                std::lock_guard<std::mutex> lock(scan_queue_mutex_);
                if (!scan_queue_.empty()) {
                    msg = scan_queue_.front();
                    scan_queue_.pop();
                }
            }

            if (msg) {
                // Process laser scan
                RCLCPP_DEBUG(this->get_logger(), "Processing scan with %zu points",
                           msg->ranges.size());
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Prevent busy waiting
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscription_;

    std::queue<sensor_msgs::msg::Image::SharedPtr> image_queue_;
    std::queue<sensor_msgs::msg::LaserScan::SharedPtr> scan_queue_;
    std::mutex image_queue_mutex_;
    std::mutex scan_queue_mutex_;

    std::thread image_processing_thread_;
    std::thread scan_processing_thread_;
    std::atomic<bool> running_{true};
};
```

## Performance Optimization

### Memory-Efficient Processing

```cpp
// Memory-efficient sensor processing
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <memory>

class MemoryEfficientProcessor : public rclcpp::Node
{
public:
    MemoryEfficientProcessor() : Node("memory_efficient_processor")
    {
        // Use small queue sizes to limit memory usage
        rclcpp::QoS qos_profile(1);  // Only keep 1 message in queue

        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "camera/image_raw", qos_profile,
            std::bind(&MemoryEfficientProcessor::image_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Memory-efficient processor initialized");
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Process image without copying data unnecessarily
        // Use in-place operations where possible

        // For example, if we only need to check image properties:
        if (msg->width > 0 && msg->height > 0) {
            // Process only essential information
            last_image_info_.width = msg->width;
            last_image_info_.height = msg->height;
            last_image_info_.encoding = msg->encoding;

            RCLCPP_DEBUG(this->get_logger(), "Processed image info: %dx%d %s",
                       msg->width, msg->height, msg->encoding.c_str());
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

## Sensor Calibration

### Calibration Considerations

```
Sensor Calibration Pipeline:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Calibration   │───▶│   Parameter     │───▶│   Correction    │
│   Target        │    │   Estimation    │    │   Application   │
│   (Chessboard,  │    │   (Intrinsic &  │    │   (Transforms,  │
│    AprilTag)     │    │    Extrinsic)   │    │    Corrections) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Calibration Validation                       │
│  - Reprojection error                                         │
│  - Accuracy metrics                                           │
│  - Consistency checks                                         │
└─────────────────────────────────────────────────────────────────┘
```

## Troubleshooting Common Issues

### Sensor Synchronization

```cpp
// Message synchronizer for multiple sensors
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

class SynchronizedSensorProcessor : public rclcpp::Node
{
public:
    SynchronizedSensorProcessor() : Node("synchronized_sensor_processor")
    {
        // Use message filters for synchronized processing
        image_sub_.subscribe(this, "camera/image_raw");
        info_sub_.subscribe(this, "camera/camera_info");

        sync_ = std::make_shared<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo>>(
            image_sub_, info_sub_, 10);
        sync_->registerCallback(std::bind(&SynchronizedSensorProcessor::callback, this,
                                        std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Synchronized processor initialized");
    }

private:
    void callback(const sensor_msgs::msg::Image::SharedPtr& image,
                 const sensor_msgs::msg::CameraInfo::SharedPtr& info)
    {
        RCLCPP_INFO(this->get_logger(), "Received synchronized image and camera info");
        // Process synchronized data
    }

    message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
    message_filters::Subscriber<sensor_msgs::msg::CameraInfo> info_sub_;
    std::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo>> sync_;
};
```

### Common Sensor Issues and Solutions

**Timing Issues:**
- Use `sensor_msgs::msg::TimeReference` for precise timing
- Implement proper message filters for synchronization
- Use hardware triggering when available

**Data Quality:**
- Implement sensor health monitoring
- Use outlier detection and filtering
- Apply appropriate noise models

**Resource Management:**
- Use appropriate QoS settings
- Implement data decimation when needed
- Use efficient data structures

## Best Practices

### Sensor Processing Guidelines

1. **Use Appropriate QoS Settings**: Match sensor data requirements (e.g., best effort for images, reliable for safety-critical data)

2. **Implement Proper Error Handling**: Check for sensor failures, data corruption, and out-of-range values

3. **Optimize Memory Usage**: Use appropriate queue sizes and avoid unnecessary data copying

4. **Consider Real-Time Requirements**: Use real-time scheduling for time-critical sensor processing

5. **Validate Data Quality**: Implement checks for sensor health and data validity

### Performance Optimization Checklist

```
Sensor Processing Optimization Checklist:
□ Use appropriate QoS settings for each sensor type
□ Implement efficient data structures and algorithms
□ Use multi-threading for independent sensor streams
□ Apply data decimation where appropriate
□ Implement proper memory management
□ Use GPU acceleration for computationally intensive tasks
□ Monitor and log performance metrics
□ Validate sensor data quality and timing
```

## Exercises and Knowledge Checkpoints

### Knowledge Checkpoints

1. What are the main sensor types used in humanoid robotics?
2. How do you handle different data rates from multiple sensors?
3. What is the difference between sensor fusion and sensor integration?
4. How do you optimize sensor processing for real-time performance?
5. What are the key considerations for sensor calibration?

### Hands-On Exercise

Create a ROS 2 node that:
1. Subscribes to camera and LiDAR data simultaneously
2. Implements a simple sensor fusion algorithm to combine the data
3. Uses multi-threading to process each sensor stream independently
4. Implements proper error handling and data validation
5. Publishes the fused information for use by other nodes

## Summary

Sensor data processing is fundamental to humanoid robotics, enabling robots to perceive and understand their environment. This chapter covered various sensor types, processing patterns, fusion techniques, and optimization strategies. Proper sensor processing requires understanding of data rates, synchronization, memory management, and real-time constraints. The combination of multiple sensors through fusion algorithms provides comprehensive environmental awareness necessary for autonomous humanoid robot operation.

## Next Steps

Continue to the next section: [ROS 2 Troubleshooting Guide](./troubleshooting-ros2.md)