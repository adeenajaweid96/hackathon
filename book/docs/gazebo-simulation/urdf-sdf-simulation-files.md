# URDF/SDF Simulation Files for Humanoid Robots

## Overview
This chapter explains the Unified Robot Description Format (URDF) and Simulation Description Format (SDF) used in Gazebo simulation. These XML-based formats define robot geometry, kinematics, dynamics, and sensor configurations essential for accurate simulation of humanoid robots.

## Learning Objectives
- Understand URDF and SDF formats and their differences
- Create URDF files for humanoid robot models
- Configure SDF files for simulation environments
- Integrate physics and sensor properties for realistic simulation
- Optimize models for simulation performance

## Prerequisites
- Understanding of ROS 2 concepts
- Basic knowledge of robot kinematics
- Experience with XML file formats
- Completed "Setting Up Gazebo Simulation Environment" chapter

## Table of Contents
1. [Introduction to URDF and SDF](#introduction-to-urdf-and-sdf)
2. [URDF Structure for Humanoid Robots](#urdf-structure-for-humanoid-robots)
3. [SDF World Files](#sdf-world-files)
4. [Physics Configuration](#physics-configuration)
5. [Sensor Integration in Simulation](#sensor-integration-in-simulation)
6. [Humanoid-Specific Considerations](#humanoid-specific-considerations)
7. [Optimization Techniques](#optimization-techniques)
8. [Troubleshooting Common Issues](#troubleshooting-common-issues)
9. [Best Practices](#best-practices)

## Introduction to URDF and SDF

### URDF (Unified Robot Description Format)
URDF is primarily used in ROS to describe robot models. It defines:
- Robot kinematic structure (links and joints)
- Physical properties (mass, inertia, visual, collision)
- Sensor locations and properties

### SDF (Simulation Description Format)
SDF is used by Gazebo to describe:
- Complete simulation environments
- Robot models with Gazebo-specific plugins
- World properties (physics, lighting, models)
- Sensor configurations with simulation-specific parameters

## URDF Structure for Humanoid Robots

### Basic URDF Structure
```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.2 0.2 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.2 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Joints define connections between links -->
  <joint name="base_to_head" type="fixed">
    <parent link="base_link"/>
    <child link="head_link"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
  </joint>

  <link name="head_link">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
</robot>
```

### Humanoid Robot URDF Example
```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base/Body -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="skin">
        <color rgba="0.8 0.6 0.4 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.004"/>
    </inertial>
  </link>

  <joint name="neck_joint" type="fixed">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_shoulder">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_shoulder"/>
    <origin xyz="-0.15 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>
</robot>
```

## SDF World Files

### Basic SDF World Structure
```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="humanoid_world">
    <!-- Physics engine configuration -->
    <physics name="default_physics" type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Robot model -->
    <model name="humanoid_robot">
      <!-- Model definition goes here -->
    </model>
  </world>
</sdf>
```

## Physics Configuration

### Material Properties
```xml
<material>
  <ambient>0.3 0.3 0.3 1</ambient>
  <diffuse>0.8 0.8 0.8 1</diffuse>
  <specular>0.1 0.1 0.1 1</specular>
  <emissive>0 0 0 1</emissive>
</material>
```

### Collision Properties
For humanoid robots, proper collision detection is crucial for realistic interaction:
```xml
<collision name="collision">
  <geometry>
    <mesh>
      <uri>model://humanoid/meshes/link_collision.stl</uri>
    </mesh>
  </geometry>
  <surface>
    <friction>
      <ode>
        <mu>1.0</mu>
        <mu2>1.0</mu2>
      </ode>
    </friction>
    <bounce>
      <restitution_coefficient>0.1</restitution_coefficient>
      <threshold>100000</threshold>
    </bounce>
  </surface>
</collision>
```

## Sensor Integration in Simulation

### Camera Sensor
```xml
<sensor name="camera" type="camera">
  <camera>
    <horizontal_fov>1.047</horizontal_fov>
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

### IMU Sensor
```xml
<sensor name="imu" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <topic>imu/data</topic>
  <visualize>false</visualize>
</sensor>
```

### LiDAR Sensor
```xml
<sensor name="lidar" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>
        <resolution>1</resolution>
        <min_angle>-1.570796</min_angle>
        <max_angle>1.570796</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <plugin filename="libgazebo_ros_ray_sensor.so" name="gazebo_ros_head_laser">
    <ros>
      <namespace>laser</namespace>
      <remapping>~/out:=scan</remapping>
    </ros>
    <output_type>sensor_msgs/LaserScan</output_type>
  </plugin>
</sensor>
```

## Humanoid-Specific Considerations

### Complete Humanoid Robot Model
Here's a more complete example of a humanoid robot model with proper kinematic chains:

```xml
<?xml version="1.0"?>
<robot name="tutorial_robot">
  <!-- Base/Fixed link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.1 0.2 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.2 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.025" iyz="0" izz="0.05"/>
    </inertial>
  </link>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.4"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3.0"/>
      <origin xyz="0 0 0.1"/>
      <inertia ixx="0.04" ixy="0" ixz="0" iyy="0.06" iyz="0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
  </joint>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="skin">
        <color rgba="0.8 0.6 0.4 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.004"/>
    </inertial>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="50" velocity="2"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_shoulder">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
      <material name="gray"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.0005"/>
    </inertial>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_shoulder"/>
    <origin xyz="-0.15 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
      <material name="gray"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.006" ixy="0" ixz="0" iyy="0.006" iyz="0" izz="0.0005"/>
    </inertial>
  </link>

  <joint name="left_upper_arm_joint" type="revolute">
    <parent link="left_shoulder"/>
    <child link="left_upper_arm"/>
    <origin xyz="0 0 -0.1" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2.0" upper="2.0" effort="50" velocity="2"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.035"/>
      </geometry>
      <material name="gray"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.035"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.6"/>
      <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.0004"/>
    </inertial>
  </link>

  <joint name="left_lower_arm_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2.0" upper="2.0" effort="50" velocity="2"/>
  </joint>

  <link name="left_hand">
    <visual>
      <geometry>
        <box size="0.1 0.08 0.05"/>
      </geometry>
      <material name="skin"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.08 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.0002" ixy="0" ixz="0" iyy="0.0003" iyz="0" izz="0.0004"/>
    </inertial>
  </link>

  <joint name="left_hand_joint" type="revolute">
    <parent link="left_lower_arm"/>
    <child link="left_hand"/>
    <origin xyz="0 0 -0.175" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="20" velocity="1"/>
  </joint>

  <!-- Right Arm (similar to left, mirrored) -->
  <link name="right_shoulder">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
      <material name="gray"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.0005"/>
    </inertial>
  </link>

  <joint name="right_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_shoulder"/>
    <origin xyz="0.15 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2"/>
  </joint>

  <!-- Additional links for right arm would continue similarly -->

  <!-- Left Leg -->
  <link name="left_hip">
    <visual>
      <geometry>
        <box size="0.08 0.12 0.08"/>
      </geometry>
      <material name="gray"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.08 0.12 0.08"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="left_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_hip"/>
    <origin xyz="-0.05 0 -0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1"/>
  </joint>

  <!-- Additional leg links would continue similarly -->
</robot>
```

### Balance and Stability
Humanoid robots require special attention to center of mass and stability:
```xml
<!-- Ensure the center of mass is low for stability -->
<inertial>
  <mass value="10.0"/>
  <origin xyz="0 0 -0.1"/> <!-- Lower center of mass -->
  <inertia ixx="0.5" ixy="0" ixz="0" iyy="0.5" iyz="0" izz="0.2"/>
</inertial>
```

### Joint Limitations
Humanoid joints should reflect human-like ranges of motion:
```xml
<joint name="hip_joint" type="revolute">
  <parent link="torso"/>
  <child link="thigh"/>
  <axis xyz="0 0 1"/>
  <limit lower="-0.52" upper="0.52" effort="100" velocity="2"/> <!-- 30 degrees -->
</joint>
```

### Gazebo-Specific Humanoid Configuration
When configuring humanoid robots for Gazebo simulation, add these Gazebo-specific tags:

```xml
<gazebo reference="left_foot">
  <mu1>1.0</mu1>
  <mu2>1.0</mu2>
  <kp>1000000.0</kp>
  <kd>100.0</kd>
  <fdir1>1 0 0</fdir1>
  <maxVel>1.0</maxVel>
  <minDepth>0.001</minDepth>
</gazebo>

<!-- Add plugins for ROS 2 control -->
<gazebo>
  <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
    <parameters>$(find my_robot_description)/config/my_robot_controllers.yaml</parameters>
  </plugin>
</gazebo>
```

### Kinematic Chain Configuration
Humanoid robots have specific kinematic chains that should be properly defined:

1. **Right and Left Arms**: Shoulder → Elbow → Wrist → Hand
2. **Right and Left Legs**: Hip → Knee → Ankle → Foot
3. **Neck-Head**: Torso → Neck → Head
4. **Torso**: Base → Torso → Head

### Mass Distribution for Humanoids
Proper mass distribution is critical for humanoid balance:

```xml
<!-- Example mass distribution for a simplified humanoid -->
<link name="torso">
  <inertial>
    <mass value="5.0"/>  <!-- 40% of total mass -->
    <origin xyz="0 0 0.1"/>
    <inertia ixx="0.08" ixy="0" ixz="0" iyy="0.12" iyz="0" izz="0.06"/>
  </inertial>
</link>

<link name="head">
  <inertial>
    <mass value="1.5"/>  <!-- 10% of total mass -->
    <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
  </inertial>
</link>

<link name="upper_arm">
  <inertial>
    <mass value="0.8"/>  <!-- 3% of total mass per arm -->
    <inertia ixx="0.003" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.0005"/>
  </inertial>
</link>
```

## Optimization Techniques

### Level of Detail (LOD)
Use different mesh complexities for visual and collision models:
```xml
<link name="optimized_link">
  <!-- Detailed mesh for visualization -->
  <visual>
    <geometry>
      <mesh>
        <uri>model://humanoid/meshes/link_visual.dae</uri>
      </mesh>
    </geometry>
  </visual>
  <!-- Simplified mesh for collision -->
  <collision>
    <geometry>
      <mesh>
        <uri>model://humanoid/meshes/link_collision_simple.stl</uri>
    </mesh>
  </collision>
</link>
```

### Model Instancing
For multiple similar robots, use instancing to reduce memory usage:
```xml
<!-- Instead of defining multiple similar robots, use parameters -->
<model name="humanoid_01">
  <!-- Use the same base model with different parameters -->
</model>
<model name="humanoid_02">
  <!-- Use the same base model with different parameters -->
</model>
```

## Troubleshooting Common Issues

### URDF Validation
Validate your URDF files:
```bash
# Check URDF syntax
check_urdf /path/to/robot.urdf

# Visualize the robot structure
urdf_to_graphiz /path/to/robot.urdf
```

### Common SDF Issues
- Ensure all mesh files are accessible and properly referenced
- Check that all required plugins are installed
- Verify coordinate frames are correctly defined
- Ensure physics properties are reasonable for your robot

### Performance Issues
- Reduce the number of triangles in collision meshes
- Use simpler geometric shapes where possible
- Limit the update rate of sensors
- Optimize physics parameters (step size, update rate)

## Best Practices

1. **Modular Design**: Break complex robots into modular components
2. **Parameterization**: Use xacro for parameterized URDFs
3. **Validation**: Always validate URDF/SDF files before simulation
4. **Realistic Physics**: Use realistic mass and inertia values
5. **Consistent Units**: Use consistent units throughout (SI units recommended)
6. **Documentation**: Comment complex URDF/SDF files for maintainability
7. **Version Control**: Track changes to URDF/SDF files with version control

## Exercises

1. Create a simple humanoid URDF model with at least 5 links
2. Define collision and visual properties for each link
3. Create an SDF world file with your humanoid robot
4. Add basic sensors (camera, IMU) to your robot model
5. Validate your URDF and test it in Gazebo

## Exercises

1. Create a simple humanoid URDF model with at least 5 links and 4 joints
2. Define collision and visual properties for each link with appropriate materials
3. Create an SDF world file that includes your humanoid robot model
4. Add a camera and IMU sensor to your robot model with proper mounting
5. Validate your URDF file using the check_urdf tool and visualize in RViz
6. Implement a basic controller for your robot's joints using ROS 2

## Next Steps

After completing this chapter, proceed to learn about Unity robot visualization to understand how to create advanced 3D visualizations for your humanoid robots.