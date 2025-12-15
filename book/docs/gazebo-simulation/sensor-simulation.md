# Sensor Simulation for Humanoid Robots in Gazebo

## Overview
This chapter covers the simulation of various sensors commonly used in humanoid robots, including cameras, LiDAR, IMUs, force/torque sensors, and other perception systems. Proper sensor simulation is crucial for developing and testing perception algorithms in a safe and cost-effective manner.

## Learning Objectives
- Understand different types of sensors used in humanoid robots
- Configure and simulate camera sensors in Gazebo
- Implement LiDAR and depth sensor simulation
- Set up IMU and inertial sensor simulation
- Simulate force/torque sensors for manipulation
- Validate sensor data quality and accuracy
- Optimize sensor simulation for performance

## Prerequisites
- Understanding of ROS 2 concepts
- Knowledge of Gazebo simulation environment
- Experience with URDF/SDF files
- Basic understanding of sensor physics and characteristics

## Table of Contents
1. [Introduction to Robot Sensors](#introduction-to-robot-sensors)
2. [Camera Sensor Simulation](#camera-sensor-simulation)
3. [LiDAR and Range Sensor Simulation](#lidar-and-range-sensor-simulation)
4. [IMU and Inertial Sensor Simulation](#imu-and-inertial-sensor-simulation)
5. [Force/Torque Sensor Simulation](#forcetorque-sensor-simulation)
6. [Depth and RGB-D Sensor Simulation](#depth-and-rgbd-sensor-simulation)
7. [Multi-Sensor Fusion in Simulation](#multi-sensor-fusion-in-simulation)
8. [Sensor Noise and Realism](#sensor-noise-and-realism)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting Common Issues](#troubleshooting-common-issues)
11. [Best Practices](#best-practices)

## Introduction to Robot Sensors

### Sensor Categories for Humanoid Robots
Humanoid robots typically use several types of sensors:

1. **Vision Sensors**: Cameras, depth sensors for environment perception
2. **Inertial Sensors**: IMUs, gyroscopes for orientation and motion
3. **Proprioceptive Sensors**: Joint encoders, force/torque sensors
4. **Range Sensors**: LiDAR, ultrasonic sensors for obstacle detection
5. **Tactile Sensors**: Force sensors for manipulation and interaction

### Sensor Simulation Challenges
- **Realism**: Simulating sensor noise and limitations
- **Performance**: Balancing accuracy with computational efficiency
- **Integration**: Ensuring sensors work with ROS 2 communication
- **Calibration**: Maintaining consistent sensor parameters

## Camera Sensor Simulation

### Basic Camera Configuration
Camera sensors in Gazebo are configured using the `<sensor>` tag in SDF files:

```xml
<sensor name="camera" type="camera">
  <camera>
    <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees in radians -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
  </camera>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
</sensor>
```

### Advanced Camera Properties
```xml
<sensor name="high_res_camera" type="camera">
  <camera>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>1280</width>
      <height>720</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.05</near>
      <far>50</far>
    </clip>
    <distortion>
      <k1>-0.177323</k1>
      <k2>0.03609</k2>
      <k3>-0.000394</k3>
      <p1>-0.000449</p1>
      <p2>0.000324</p2>
      <center>0.5 0.5</center>
    </distortion>
  </camera>
  <plugin filename="libgazebo_ros_camera.so" name="camera_controller">
    <ros>
      <namespace>camera</namespace>
      <remapping>~/image_raw:=image</remapping>
      <remapping>~/camera_info:=camera_info</remapping>
    </ros>
    <camera_name>front_camera</camera_name>
    <frame_name>camera_frame</frame_name>
    <hack_baseline>0.07</hack_baseline>
    <distortion_k1>-0.177323</distortion_k1>
    <distortion_k2>0.03609</distortion_k2>
    <distortion_k3>-0.000394</distortion_k3>
    <distortion_t1>-0.000449</distortion_t1>
    <distortion_t2>0.000324</distortion_t2>
  </plugin>
</sensor>
```

### Stereo Camera Setup
For depth perception, configure stereo cameras:

```xml
<!-- Left camera -->
<sensor name="left_camera" type="camera">
  <!-- ... camera configuration ... -->
  <pose>0.05 0 0 0 0 0</pose>  <!-- Offset from center -->
</sensor>

<!-- Right camera -->
<sensor name="right_camera" type="camera">
  <!-- ... camera configuration ... -->
  <pose>-0.05 0 0 0 0 0</pose>  <!-- Offset from center -->
</sensor>
```

### Camera Mounting on Humanoid Robot
For humanoid robots, cameras are typically mounted on the head or torso:

```xml
<link name="head_camera_link">
  <visual>
    <geometry>
      <box size="0.02 0.02 0.02"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <box size="0.02 0.02 0.02"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.01"/>
    <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
  </inertial>
</link>

<joint name="head_camera_joint" type="fixed">
  <parent link="head"/>
  <child link="head_camera_link"/>
  <origin xyz="0.05 0 0.05" rpy="0 0 0"/>  <!-- Positioned on head -->
</joint>

<sensor name="head_camera" type="camera">
  <pose>0 0 0 0 0 0</pose>
  <camera>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
  </camera>
</sensor>
```

## LiDAR and Range Sensor Simulation

### 2D LiDAR Configuration
```xml
<sensor name="laser_2d" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>
        <resolution>1</resolution>
        <min_angle>-1.570796</min_angle>  <!-- -90 degrees -->
        <max_angle>1.570796</max_angle>   <!-- 90 degrees -->
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <update_rate>10</update_rate>
  <plugin filename="libgazebo_ros_ray_sensor.so" name="gazebo_ros_head_hokuyo">
    <ros>
      <namespace>laser</namespace>
      <remapping>~/out:=scan</remapping>
    </ros>
    <output_type>sensor_msgs/LaserScan</output_type>
    <frame_name>laser_frame</frame_name>
  </plugin>
</sensor>
```

### 3D LiDAR Configuration
For humanoid robots requiring 3D perception:

```xml
<sensor name="velodyne_vlp16" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>1800</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
      <vertical>
        <samples>16</samples>
        <resolution>1</resolution>
        <min_angle>-0.261799</min_angle>  <!-- -15 degrees -->
        <max_angle>0.261799</max_angle>   <!-- 15 degrees -->
      </vertical>
    </scan>
    <range>
      <min>0.3</min>
      <max>100.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <plugin filename="libgazebo_ros_velodyne_gpu_laser.so" name="gazebo_ros_head_velodyne">
    <topic_name>velodyne_points</topic_name>
    <frame_name>velodyne_frame</frame_name>
    <min_range>0.3</min_range>
    <max_range>100.0</max_range>
    <gaussian_noise>0.01</gaussian_noise>
  </plugin>
</sensor>
```

### Ultrasonic Sensor Simulation
For close-range obstacle detection:

```xml
<sensor name="ultrasonic_sensor" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>1</samples>
        <resolution>1</resolution>
        <min_angle>0</min_angle>
        <max_angle>0.1</max_angle>  <!-- Narrow beam -->
      </horizontal>
    </scan>
    <range>
      <min>0.02</min>
      <max>4.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <update_rate>10</update_rate>
</sensor>
```

## IMU and Inertial Sensor Simulation

### IMU Configuration
```xml
<sensor name="imu_sensor" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.0017</stddev>  <!-- ~0.1 deg/s (1-sigma) -->
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.0017</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.0017</stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>  <!-- 1-sigma noise density: 17 mg */
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <plugin filename="libgazebo_ros_imu.so" name="imu_plugin">
    <ros>
      <namespace>imu</namespace>
      <remapping>~/out:=data</remapping>
    </ros>
    <frame_name>imu_frame</frame_name>
    <body_name>torso</body_name>
    <update_rate>100</update_rate>
    <gaussian_noise>0.0017</gaussian_noise>
  </plugin>
</sensor>
```

### Multiple IMU Placement
For humanoid robots, IMUs are often placed on multiple body parts:

```xml
<!-- Torso IMU -->
<sensor name="torso_imu" type="imu">
  <!-- ... configuration ... -->
  <pose>0 0 0 0 0 0</pose>
</sensor>

<!-- Head IMU -->
<sensor name="head_imu" type="imu">
  <!-- ... configuration ... -->
  <pose>0 0 0.3 0 0 0</pose>  <!-- Positioned at head -->
</sensor>

<!-- Foot IMU (for balance) -->
<sensor name="left_foot_imu" type="imu">
  <!-- ... configuration ... -->
  <pose>0 0.1 -0.5 0 0 0</pose>  <!-- Positioned at left foot -->
</sensor>
```

## Force/Torque Sensor Simulation

### Joint Force/Torque Sensors
```xml
<sensor name="left_ankle_ft" type="force_torque">
  <force_torque>
    <frame>child</frame>
    <measure_direction>child_to_parent</measure_direction>
  </force_torque>
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <plugin filename="libgazebo_ros_ft_sensor.so" name="left_ankle_ft_plugin">
    <ros>
      <namespace>ft_sensors</namespace>
      <remapping>~/out:=left_ankle_wrench</remapping>
    </ros>
    <frame_name>left_ankle_frame</frame_name>
    <topic_name>left_ankle_wrench</topic_name>
  </plugin>
</sensor>
```

### Gripper Force Sensors
For manipulation tasks:

```xml
<sensor name="gripper_force_sensor" type="force_torque">
  <force_torque>
    <frame>child</frame>
    <measure_direction>child_to_parent</measure_direction>
  </force_torque>
  <always_on>true</always_on>
  <update_rate>100</update_rate>
</sensor>
```

## Depth and RGB-D Sensor Simulation

### Depth Camera Configuration
```xml
<sensor name="depth_camera" type="depth">
  <camera>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
  </camera>
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
  <plugin filename="libgazebo_ros_openni_kinect.so" name="kinect_camera">
    <alwaysOn>true</alwaysOn>
    <updateRate>10</updateRate>
    <cameraName>kinect2</cameraName>
    <imageTopicName>/rgb/image_raw</imageTopicName>
    <depthImageTopicName>/depth/image_raw</depthImageTopicName>
    <pointCloudTopicName>/depth/points</pointCloudTopicName>
    <cameraInfoTopicName>/rgb/camera_info</cameraInfoTopicName>
    <depthImageCameraInfoTopicName>/depth/camera_info</depthImageCameraInfoTopicName>
    <frameName>kinect2_rgb_optical_frame</frameName>
    <baseline>0.1</baseline>
    <distortion_k1>0.0</distortion_k1>
    <distortion_k2>0.0</distortion_k2>
    <distortion_k3>0.0</distortion_k3>
    <distortion_t1>0.0</distortion_t1>
    <distortion_t2>0.0</distortion_t2>
    <pointCloudCutoff>0.1</pointCloudCutoff>
    <pointCloudCutoffMax>3.0</pointCloudCutoffMax>
    <CxPrime>0.0</CxPrime>
    <Cx>0.0</Cx>
    <Cy>0.0</Cy>
    <focalLength>0.0</focalLength>
    <hackBaseline>0.0</hackBaseline>
  </plugin>
</sensor>
```

## Multi-Sensor Fusion in Simulation

### Sensor Data Synchronization
For humanoid robots, multiple sensors need to be synchronized:

```xml
<!-- Use the same timestamp for related sensors -->
<sensor name="camera" type="camera">
  <!-- ... camera config ... -->
  <update_rate>30</update_rate>
</sensor>

<sensor name="imu" type="imu">
  <!-- ... imu config ... -->
  <update_rate>30</update_rate>  <!-- Match camera rate -->
</sensor>
```

### Sensor Integration in Control Systems
Example of using multiple sensors for humanoid balance:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Vector3
import numpy as np

class HumanoidSensorFusion(Node):
    def __init__(self):
        super().__init__('humanoid_sensor_fusion')

        # Subscribe to multiple sensors
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)

        # Publishers for fused data
        self.balance_pub = self.create_publisher(
            Vector3, '/balance_state', 10)

        self.imu_data = None
        self.joint_data = None

    def imu_callback(self, msg):
        self.imu_data = msg
        self.update_balance_state()

    def joint_callback(self, msg):
        self.joint_data = msg
        self.update_balance_state()

    def update_balance_state(self):
        if self.imu_data and self.joint_data:
            # Calculate balance based on IMU and joint data
            roll, pitch, yaw = self.quaternion_to_euler(
                self.imu_data.orientation)

            # Publish balance state for controller
            balance_msg = Vector3()
            balance_msg.x = roll  # Roll angle
            balance_msg.y = pitch  # Pitch angle
            balance_msg.z = 0.0    # For ZMP calculation

            self.balance_pub.publish(balance_msg)

    def quaternion_to_euler(self, q):
        # Convert quaternion to Euler angles
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (q.w * q.y - q.z * q.x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

def main(args=None):
    rclpy.init(args=args)
    sensor_fusion = HumanoidSensorFusion()
    rclpy.spin(sensor_fusion)
    sensor_fusion.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor Noise and Realism

### Adding Realistic Noise
```xml
<sensor name="realistic_camera" type="camera">
  <camera>
    <!-- ... camera config ... -->
  </camera>
  <noise>
    <type>gaussian</type>
    <mean>0.0</mean>
    <stddev>0.007</stddev>  <!-- Add realistic noise -->
  </noise>
</sensor>
```

### Environmental Effects
Simulate environmental conditions affecting sensors:

```xml
<!-- In world file, add atmospheric effects -->
<scene>
  <ambient>0.4 0.4 0.4 1</ambient>
  <background>0.7 0.7 0.7 1</background>
  <shadows>true</shadows>
  <grid>false</grid>
  <origin_visual>false</origin_visual>
</scene>
```

## Performance Optimization

### Sensor Update Rate Management
Balance accuracy with performance:

```xml
<!-- High priority sensors (balance) -->
<sensor name="imu" type="imu">
  <update_rate>200</update_rate>  <!-- High rate for balance -->
</sensor>

<!-- Medium priority sensors -->
<sensor name="camera" type="camera">
  <update_rate>30</update_rate>   <!-- Standard rate -->
</sensor>

<!-- Low priority sensors -->
<sensor name="gps" type="gps">
  <update_rate>10</update_rate>   <!-- Lower rate for GPS -->
</sensor>
```

### Multi-Threading for Sensor Processing
```python
#!/usr/bin/env python3
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu

class OptimizedSensorProcessor(Node):
    def __init__(self):
        super().__init__('optimized_sensor_processor')

        # Separate threads for different sensor types
        self.camera_thread = threading.Thread(target=self.process_camera)
        self.imu_thread = threading.Thread(target=self.process_imu)

        # Initialize subscribers
        self.camera_sub = self.create_subscription(
            Image, '/camera/image', self.camera_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)

        self.camera_data = None
        self.imu_data = None
        self.lock = threading.Lock()

    def camera_callback(self, msg):
        with self.lock:
            self.camera_data = msg

    def imu_callback(self, msg):
        with self.lock:
            self.imu_data = msg

def main(args=None):
    rclpy.init(args=args)
    processor = OptimizedSensorProcessor()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Troubleshooting Common Issues

### Sensor Not Publishing Data
- Check that the sensor plugin is properly loaded
- Verify ROS topic names and namespaces
- Ensure the sensor is properly attached to a link
- Check Gazebo simulation is running

### Performance Issues
- Reduce sensor update rates
- Simplify sensor models if using complex meshes
- Limit the number of active sensors
- Use appropriate image resolutions

### Calibration Problems
- Verify sensor frame transformations
- Check sensor mounting positions
- Validate intrinsic and extrinsic parameters
- Ensure proper coordinate frame conventions

## Best Practices

1. **Sensor Placement**: Position sensors realistically on the robot
2. **Update Rates**: Match sensor update rates to application needs
3. **Noise Modeling**: Include realistic noise models for robust algorithms
4. **Frame Conventions**: Use consistent coordinate frame conventions
5. **Validation**: Regularly validate sensor data against real hardware
6. **Documentation**: Document sensor specifications and limitations
7. **Testing**: Test sensor fusion algorithms with simulated data

## Exercises

1. Configure a camera sensor on a humanoid robot model
2. Set up an IMU for balance control simulation
3. Implement a 2D LiDAR for obstacle detection
4. Create a multi-sensor fusion node for humanoid perception
5. Add realistic noise models to your sensors
6. Optimize sensor update rates for performance

## Synthetic Data Generation for AI Training

### Overview
Synthetic data generation is the process of creating artificial data using simulation environments rather than collecting it from the real world. For humanoid robotics, synthetic data is crucial for training AI models, especially perception and control systems, as it allows for rapid data collection with perfect ground truth labels.

### Benefits of Synthetic Data
- **Cost-effective**: No need for expensive data collection campaigns
- **Ground truth**: Perfect annotations for training data
- **Safety**: Train dangerous behaviors in simulation
- **Scalability**: Generate unlimited data samples
- **Control**: Create specific scenarios and edge cases
- **Diversity**: Simulate various environments and conditions

### Types of Synthetic Data for Humanoid Robots

#### Vision Data
- **RGB Images**: Natural images for object recognition
- **Depth Maps**: Depth information for 3D understanding
- **Semantic Segmentation**: Pixel-level object classification
- **Instance Segmentation**: Individual object instance labeling
- **Optical Flow**: Motion information between frames

#### Sensor Data
- **LiDAR Point Clouds**: 3D environment representation
- **IMU Readings**: Inertial measurement data
- **Force/Torque Data**: Contact force information
- **Joint Position Data**: Robot configuration information

### Domain Randomization

Domain randomization is a technique to make synthetic data more transferable to the real world by randomizing various aspects of the simulation:

```python
#!/usr/bin/env python3
# Example of domain randomization for synthetic data generation
import random
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization parameters"""
    lighting_range: Tuple[float, float] = (0.5, 2.0)  # Light intensity range
    texture_probabilities: dict = None  # Texture randomization
    camera_noise: Tuple[float, float] = (0.0, 0.01)  # Gaussian noise parameters
    background_objects: list = None  # Random background objects
    occlusion_probability: float = 0.3  # Probability of partial occlusion

class SyntheticDataGenerator:
    def __init__(self, config: DomainRandomizationConfig):
        self.config = config
        self.step_count = 0

    def randomize_lighting(self):
        """Randomize lighting conditions in the scene"""
        light_intensity = random.uniform(
            self.config.lighting_range[0],
            self.config.lighting_range[1]
        )

        light_color = [
            random.uniform(0.8, 1.2),  # Red
            random.uniform(0.8, 1.2),  # Green
            random.uniform(0.8, 1.2)   # Blue
        ]

        return light_intensity, light_color

    def randomize_textures(self, available_textures: list):
        """Randomize surface textures in the environment"""
        if not self.config.texture_probabilities:
            return random.choice(available_textures)

        # Weighted random selection based on probabilities
        textures = list(self.config.texture_probabilities.keys())
        weights = list(self.config.texture_probabilities.values())
        return random.choices(textures, weights=weights)[0]

    def add_camera_noise(self, image):
        """Add realistic noise to camera images"""
        noise_mean, noise_std = self.config.camera_noise
        noise = np.random.normal(noise_mean, noise_std, image.shape)
        noisy_image = np.clip(image + noise, 0, 255)
        return noisy_image.astype(np.uint8)

    def generate_training_sample(self, robot_state, environment_state):
        """Generate a complete training sample with annotations"""
        # Apply domain randomization
        light_intensity, light_color = self.randomize_lighting()
        # Apply lighting changes to simulation here

        # Capture sensor data
        rgb_image = self.capture_rgb_image()
        depth_image = self.capture_depth_image()
        segmentation_mask = self.generate_segmentation_mask()

        # Add noise to simulate real sensors
        rgb_image = self.add_camera_noise(rgb_image)

        # Create training sample with ground truth
        training_sample = {
            'rgb_image': rgb_image,
            'depth_image': depth_image,
            'segmentation_mask': segmentation_mask,
            'robot_state': robot_state,
            'environment_state': environment_state,
            'domain_params': {
                'light_intensity': light_intensity,
                'light_color': light_color
            }
        }

        return training_sample

    def capture_rgb_image(self):
        """Simulate RGB camera capture"""
        # This would interface with the simulation environment
        # Return a numpy array representing the captured image
        pass

    def capture_depth_image(self):
        """Simulate depth camera capture"""
        # This would interface with the simulation environment
        # Return a numpy array representing the depth image
        pass

    def generate_segmentation_mask(self):
        """Generate semantic segmentation mask"""
        # This would create pixel-level object classification
        # Return a numpy array with object class labels
        pass

# Example usage
def main():
    config = DomainRandomizationConfig(
        lighting_range=(0.3, 3.0),
        camera_noise=(0.0, 0.02),
        occlusion_probability=0.25
    )

    generator = SyntheticDataGenerator(config)

    # Generate multiple samples
    for i in range(1000):  # Generate 1000 training samples
        robot_state = get_robot_state()  # Get current robot state
        env_state = get_environment_state()  # Get current environment state

        sample = generator.generate_training_sample(robot_state, env_state)

        # Save the sample to dataset
        save_training_sample(sample, f"dataset/sample_{i:06d}.npz")

def get_robot_state():
    """Get current robot state from simulation"""
    # Implementation would interface with the simulator
    pass

def get_environment_state():
    """Get current environment state from simulation"""
    # Implementation would interface with the simulator
    pass

def save_training_sample(sample, filename):
    """Save training sample to disk"""
    # Implementation would save the sample in appropriate format
    pass

if __name__ == '__main__':
    main()
```

### Synthetic Data Pipeline

Creating a complete synthetic data generation pipeline involves several components:

#### 1. Scenario Generation
```python
class ScenarioGenerator:
    """Generate diverse scenarios for synthetic data"""

    def __init__(self):
        self.scenarios = [
            "indoor_office",
            "outdoor_park",
            "warehouse",
            "home_environment",
            "cluttered_room"
        ]

    def generate_scenario(self, scenario_type: str):
        """Generate a specific scenario with randomized elements"""
        if scenario_type == "indoor_office":
            return self._create_office_environment()
        elif scenario_type == "outdoor_park":
            return self._create_park_environment()
        # ... other scenarios

    def _create_office_environment(self):
        """Create a randomized office environment"""
        # Place furniture randomly
        furniture_positions = self.randomize_furniture_placement()

        # Set lighting conditions
        lighting = self.randomize_office_lighting()

        # Add random objects
        objects = self.randomize_objects()

        return {
            'furniture': furniture_positions,
            'lighting': lighting,
            'objects': objects
        }
```

#### 2. Data Annotation
```python
class DataAnnotator:
    """Generate ground truth annotations for synthetic data"""

    def annotate_object_detection(self, scene):
        """Generate bounding box annotations"""
        annotations = []
        for obj in scene.visible_objects:
            bbox = self.calculate_bounding_box(obj)
            annotations.append({
                'object_class': obj.class_name,
                'bbox': bbox,
                'confidence': 1.0  # Perfect confidence in simulation
            })
        return annotations

    def annotate_pose_estimation(self, robot):
        """Generate pose annotations for robot parts"""
        pose_annotations = {}
        for link_name, link_pose in robot.link_poses.items():
            pose_annotations[link_name] = {
                'position': link_pose.position,
                'orientation': link_pose.orientation,
                'visibility': self.calculate_visibility(link_pose)
            }
        return pose_annotations
```

#### 3. Data Storage and Management
```python
import h5py
import numpy as np

class SyntheticDatasetManager:
    """Manage storage and access of synthetic datasets"""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset_file = None

    def create_dataset(self, num_samples: int):
        """Create a new dataset file"""
        self.dataset_file = h5py.File(self.dataset_path, 'w')

        # Create datasets for different data types
        self.rgb_dataset = self.dataset_file.create_dataset(
            'rgb_images',
            (num_samples, 480, 640, 3),
            dtype='uint8',
            chunks=True
        )

        self.depth_dataset = self.dataset_file.create_dataset(
            'depth_images',
            (num_samples, 480, 640),
            dtype='float32',
            chunks=True
        )

        self.annotations_dataset = self.dataset_file.create_dataset(
            'annotations',
            (num_samples,),
            dtype=h5py.string_dtype(),
            chunks=True
        )

    def add_sample(self, index: int, rgb_image, depth_image, annotations):
        """Add a sample to the dataset"""
        self.rgb_dataset[index] = rgb_image
        self.depth_dataset[index] = depth_image
        self.annotations_dataset[index] = str(annotations)

    def close(self):
        """Close the dataset file"""
        if self.dataset_file:
            self.dataset_file.close()
```

### Real-to-Sim Transfer Techniques

#### Texture Randomization
Randomizing textures helps models generalize better:

```python
def randomize_texture(material, randomization_params):
    """Apply texture randomization to a material"""
    # Change base color with random hue variation
    base_color = material.base_color
    hue_shift = random.uniform(-0.1, 0.1)
    material.base_color = shift_hue(base_color, hue_shift)

    # Randomize roughness
    material.roughness = random.uniform(0.1, 0.9)

    # Randomize metallic properties
    material.metallic = random.uniform(0.0, 0.2)

    return material
```

#### Sensor Noise Modeling
Adding realistic noise to synthetic data:

```python
def add_realistic_noise(image, sensor_type='camera'):
    """Add sensor-specific noise to synthetic images"""
    if sensor_type == 'camera':
        # Add photon noise (proportional to signal)
        photon_noise = np.random.poisson(image / 255.0) * 255.0
        # Add read noise (constant)
        read_noise = np.random.normal(0, 2, image.shape)
        noisy_image = image + photon_noise + read_noise
    elif sensor_type == 'depth':
        # Add depth-specific noise (increases with distance)
        depth_noise = np.random.normal(0, 0.01 * image, image.shape)
        noisy_image = image + depth_noise

    return np.clip(noisy_image, 0, 255).astype(np.uint8)
```

### Quality Assurance for Synthetic Data

#### Data Validation
```python
class SyntheticDataValidator:
    """Validate quality and consistency of synthetic data"""

    def validate_sample(self, sample):
        """Validate a single synthetic data sample"""
        issues = []

        # Check image dimensions
        if sample['rgb_image'].shape != (480, 640, 3):
            issues.append("Incorrect image dimensions")

        # Check depth validity
        if np.any(sample['depth_image'] < 0) or np.any(sample['depth_image'] > 100):
            issues.append("Invalid depth values")

        # Check annotation consistency
        if not self.check_annotation_consistency(sample):
            issues.append("Inconsistent annotations")

        return len(issues) == 0, issues

    def check_annotation_consistency(self, sample):
        """Check if annotations match the image content"""
        # Implementation would verify that bounding boxes
        # align with objects in the image
        pass
```

### Best Practices for Synthetic Data Generation

1. **Start Simple**: Begin with basic scenarios before adding complexity
2. **Validate Transfer**: Test model performance on real data regularly
3. **Monitor Distribution**: Ensure synthetic data distribution matches real data
4. **Document Parameters**: Keep track of all randomization parameters
5. **Quality Control**: Implement validation checks for generated data
6. **Progressive Complexity**: Gradually increase scene complexity
7. **Real Data Mixing**: Combine synthetic and real data for training

### Performance Considerations

#### Batch Generation
Generate data in batches for efficiency:

```python
def generate_batch(batch_size: int, generator: SyntheticDataGenerator):
    """Generate a batch of synthetic data samples"""
    batch = []
    for _ in range(batch_size):
        sample = generator.generate_training_sample()
        batch.append(sample)
    return batch
```

#### Parallel Processing
Use multiple simulation instances:

```python
from multiprocessing import Pool
import os

def parallel_data_generation(num_processes: int, samples_per_process: int):
    """Generate data using multiple parallel processes"""
    with Pool(processes=num_processes) as pool:
        args = [(samples_per_process,) for _ in range(num_processes)]
        results = pool.starmap(generate_samples_process, args)

    return results

def generate_samples_process(num_samples: int):
    """Generate samples in a separate process"""
    generator = SyntheticDataGenerator(config)
    samples = []

    for i in range(num_samples):
        sample = generator.generate_training_sample()
        samples.append(sample)

    return samples
```

## Exercises

1. Configure a 2D LiDAR sensor on a humanoid robot model and visualize the scan data
2. Set up an IMU sensor with realistic noise parameters for balance control applications
3. Implement a depth camera with proper calibration parameters and visualize point clouds
4. Create a multi-sensor fusion node that combines data from camera, IMU, and LiDAR
5. Add realistic noise models to your sensors and compare performance with clean data
6. Generate a small synthetic dataset using domain randomization techniques
7. Implement a contact sensor on a robot's foot to detect ground contact
8. Optimize sensor update rates for a humanoid robot with 20+ sensors

## Next Steps

After completing this chapter, you will have a comprehensive understanding of sensor simulation and synthetic data generation for humanoid robots. Proceed to learn about advanced simulation techniques and integration with real-world robotics applications.