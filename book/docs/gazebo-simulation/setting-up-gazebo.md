# Setting Up Gazebo Simulation Environment

## Overview
This chapter covers the setup and configuration of Gazebo simulation environment for humanoid robotics development. Gazebo provides a realistic 3D simulation environment that allows testing of robotic algorithms without physical hardware, making it essential for rapid prototyping and development.

## Learning Objectives
- Install and configure Gazebo simulation environment
- Understand Gazebo's architecture and components
- Learn to launch basic simulation worlds
- Configure sensors and physics properties
- Integrate with ROS 2 for robot simulation

## Prerequisites
- Ubuntu 22.04 LTS (recommended)
- ROS 2 Humble Hawksbill installed
- Basic understanding of Linux command line
- Familiarity with ROS 2 concepts (covered in Part II)

## Table of Contents
1. [Installation and Setup](#installation-and-setup)
2. [Understanding Gazebo Architecture](#understanding-gazebo-architecture)
3. [Launching Basic Worlds](#launching-basic-worlds)
4. [Configuring Physics Properties](#configuring-physics-properties)
5. [Sensor Integration](#sensor-integration)
6. [ROS 2 Integration](#ros-2-integration)
7. [Troubleshooting Common Issues](#troubleshooting-common-issues)
8. [Best Practices](#best-practices)

## Installation and Setup

### System Requirements
- CPU: Multi-core processor (Intel i5 or equivalent)
- RAM: 8GB minimum, 16GB recommended
- GPU: OpenGL 2.1 compatible graphics card with dedicated VRAM (2GB+ recommended)
- Storage: 5GB free space for Gazebo and models

### Installing Gazebo Garden
Gazebo Garden is the latest version of Gazebo at the time of writing. Install it using the following commands:

```bash
# Add the OSRF repository
sudo apt update && sudo apt install wget
wget https://packages.osrfoundation.org/gazebo.gpg -O /tmp/gazebo.gpg
sudo cp /tmp/gazebo.gpg /usr/share/keyrings/
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gazebo.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# Update package list
sudo apt update

# Install Gazebo Garden
sudo apt install gz-garden
```

### Installing ROS 2 Gazebo Packages
```bash
# Install ROS 2 Humble packages for Gazebo integration
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control ros-humble-gazebo-dev
```

### Verifying Installation
Test the installation by launching Gazebo:
```bash
gz sim
```

If Gazebo GUI opens successfully, your installation is complete.

## Understanding Gazebo Architecture

Gazebo consists of several key components:

### 1. Gazebo Simulator (gz-sim)
The core simulation engine responsible for physics simulation, rendering, and sensor simulation.

### 2. Gazebo Client (gz-sim-ui)
The graphical user interface that allows interaction with the simulation environment.

### 3. Model Database
A collection of pre-built 3D models for robots, objects, and environments available at https://app.gazebosim.org/fuel

### 4. Plugins System
Extensible plugin architecture allowing custom simulation behaviors and integrations.

## Launching Basic Worlds

### Default World
Start with the default world to familiarize yourself with the interface:
```bash
gz sim -v 4
```

### Custom Worlds
Gazebo supports custom world files in SDF format. Create a simple world file:

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="basic_world">
    <!-- Include a ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include a sky -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Add a simple box -->
    <model name="box">
      <pose>0 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

Save this as `basic_world.sdf` and launch it:
```bash
gz sim basic_world.sdf
```

## Configuring Physics Properties

Gazebo's physics engine is based on ignition-physics. The default physics engine is DART (Dynamic Animation and Robotics Toolkit), though ODE (Open Dynamics Engine) and Bullet are also supported.

### Physics Configuration
Physics parameters can be configured in world files:

```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
</physics>
```

Key parameters:
- `max_step_size`: Simulation time step (smaller = more accurate but slower)
- `real_time_factor`: Target simulation speed (1.0 = real-time)
- `real_time_update_rate`: Updates per second

### Physics Engine Selection
Different physics engines have different strengths:

- **ODE**: Good for general-purpose simulation, widely tested
- **DART**: Better for articulated bodies and humanoid robots
- **Bullet**: Fast for simple collision detection

```xml
<!-- Using DART for humanoid robots -->
<physics type="dart">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
</physics>
```

### Advanced Physics Parameters
For humanoid robots, you may need to tune additional parameters:

```xml
<physics type="dart">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
  <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

## Collision Handling

### Collision Detection
Gazebo uses multiple collision detection algorithms depending on the physics engine. For humanoid robots, proper collision handling is crucial for realistic interaction with the environment.

### Collision Properties
Configure collision properties for each link:

```xml
<link name="foot_link">
  <collision name="collision">
    <geometry>
      <box>
        <size>0.15 0.1 0.05</size>
      </box>
    </geometry>
    <surface>
      <friction>
        <ode>
          <mu>1.0</mu>
          <mu2>1.0</mu2>
          <fdir1>0 0 0</fdir1>
          <slip1>0</slip1>
          <slip2>0</slip2>
        </ode>
        <torsional>
          <coefficient>1.0</coefficient>
          <use_patch_radius>1</use_patch_radius>
          <surface_radius>0.001</surface_radius>
          <patch_radius>0.001</patch_radius>
          <constraint_resolution>1</constraint_resolution>
        </torsional>
      </friction>
      <bounce>
        <restitution_coefficient>0.01</restitution_coefficient>
        <threshold>100000</threshold>
      </bounce>
      <contact>
        <collide_without_contact>0</collide_without_contact>
        <collide_without_contact_bitmask>1</collide_without_contact_bitmask>
        <collide_bitmask>1</collide_bitmask>
        <ode>
          <soft_cfm>0</soft_cfm>
          <soft_erp>0.2</soft_erp>
          <kp>1000000000000</kp>
          <kd>1</kd>
          <max_vel>100</max_vel>
          <min_depth>0</min_depth>
        </ode>
        <bullet>
          <split_impulse>1</split_impulse>
          <split_impulse_penetration_threshold>-0.01</split_impulse_penetration_threshold>
          <soft_cfm>0</soft_cfm>
          <soft_erp>0.2</soft_erp>
          <kp>1000000000000</kp>
          <kd>1</kd>
          <max_vel>100</max_vel>
          <min_depth>0</min_depth>
        </bullet>
      </contact>
    </surface>
  </collision>
</link>
```

### Collision Meshes for Humanoid Robots
For humanoid robots, it's important to use appropriate collision meshes:

1. **Simplified Geometry**: Use simple shapes (boxes, cylinders, spheres) for collision to improve performance
2. **Proper Sizing**: Ensure collision meshes are slightly larger than visual meshes to prevent interpenetration
3. **Convex Hulls**: For complex shapes, use convex hulls rather than concave meshes

```xml
<link name="humanoid_arm">
  <collision>
    <geometry>
      <cylinder>
        <radius>0.04</radius>
        <length>0.3</length>
      </cylinder>
    </geometry>
  </collision>
  <visual>
    <!-- More detailed visual mesh -->
    <geometry>
      <mesh>
        <uri>meshes/arm_visual.dae</uri>
      </mesh>
    </geometry>
  </visual>
</link>
```

### Contact Sensors
For humanoid robots, you may want to add contact sensors to detect when feet touch the ground:

```xml
<sensor name="left_foot_contact" type="contact">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <contact>
    <collision>left_foot_collision</collision>
  </contact>
  <plugin filename="libgazebo_ros_contact.so" name="left_foot_contact_plugin">
    <ros>
      <namespace>contact_sensors</namespace>
      <remapping>~/out:=left_foot_contact</remapping>
    </ros>
    <frame_name>left_foot</frame_name>
    <topic_name>left_foot_contact</topic_name>
  </plugin>
</sensor>
```

### Physics Performance Optimization
For humanoid robots with many joints, consider these optimizations:

1. **Reduce update rate** for less critical physics interactions
2. **Use fixed joints** instead of high-stiffness revolute joints where appropriate
3. **Adjust solver parameters** for better stability
4. **Simplify collision meshes** for links that don't require precise collision detection

### Troubleshooting Physics Issues
Common physics problems with humanoid robots:

1. **Jittery movement**: Reduce `max_step_size` or increase solver iterations
2. **Unstable balance**: Check mass distribution and inertial properties
3. **Interpenetration**: Increase stiffness parameters or simplify collision geometry
4. **Performance issues**: Reduce the number of contact points or simplify meshes

## Sensor Integration

Gazebo supports various sensors including cameras, LiDAR, IMUs, and force/torque sensors. Sensors are defined in robot URDF/SDF files and publish data to ROS 2 topics.

### Example: Adding a Camera Sensor
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
</sensor>
```

## ROS 2 Integration

### Using gazebo_ros_pkgs
The `gazebo_ros_pkgs` provide the bridge between Gazebo and ROS 2. Common plugins include:
- `libgazebo_ros_init.so`: Initializes ROS 2 communication
- `libgazebo_ros_factory.so`: Spawns models via ROS 2 services
- `libgazebo_ros_force.so`: Applies forces via ROS 2 topics

### Launching with ROS 2
To launch Gazebo with ROS 2 integration:
```bash
# Terminal 1
ros2 launch gazebo_ros gazebo.launch.py

# Terminal 2
# Spawn a robot model
ros2 run gazebo_ros spawn_entity.py -entity my_robot -file /path/to/robot.urdf
```

## Troubleshooting Common Issues

### Graphics Issues
- If experiencing rendering problems, ensure your graphics drivers are up to date
- For NVIDIA GPUs, install proprietary drivers: `sudo apt install nvidia-driver-XXX`
- Try running with software rendering: `export LIBGL_ALWAYS_SOFTWARE=1`

### Performance Issues
- Reduce `max_step_size` for better accuracy at the cost of performance
- Lower `real_time_update_rate` to reduce computational load
- Simplify collision meshes for complex models

### Plugin Loading Issues
- Verify plugin libraries exist and are accessible
- Check ROS 2 environment is sourced: `source /opt/ros/humble/setup.bash`

## Best Practices

1. **Model Optimization**: Use simplified collision meshes separate from visual meshes
2. **World Design**: Start with simple worlds and gradually add complexity
3. **Physics Tuning**: Balance accuracy and performance based on your application needs
4. **Testing**: Regularly test simulation behavior against real-world expectations
5. **Documentation**: Maintain clear documentation of world configurations and robot models

## Exercises

1. Install Gazebo Garden and verify the installation
2. Create a custom world file with multiple objects
3. Configure physics parameters for a specific application
4. Launch a simple robot model in Gazebo and visualize sensor data

## Exercises

1. Install Gazebo Garden on your development machine and verify the installation by launching the GUI
2. Create a custom world file with at least 3 different objects and launch it in Gazebo
3. Configure physics parameters for a humanoid robot simulation with a 500Hz update rate
4. Set up a basic camera sensor on a simple robot model and visualize the output
5. Create a launch file that starts Gazebo with your custom world and robot model

## Next Steps

After completing this chapter, proceed to learn about URDF/SDF simulation files to understand how to properly configure robots for simulation.