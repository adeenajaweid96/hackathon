---
sidebar_position: 2.4
title: "URDF Robot Description"
---

# URDF Robot Description

## Overview

Unified Robot Description Format (URDF) is an XML format used in ROS to describe robot models. URDF defines the physical and visual properties of a robot, including its links, joints, inertial properties, and visual appearance. This chapter covers the fundamentals of creating and working with URDF models for humanoid robots.

## URDF Structure and Components

### Basic URDF Structure

A URDF file follows this basic structure:

```xml
<?xml version="1.0"?>
<robot name="my_humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Joints connect links -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso_link"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
  </joint>

  <link name="torso_link">
    <visual>
      <geometry>
        <cylinder radius="0.15" length="0.6"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.15" length="0.6"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.1"/>
    </inertial>
  </link>
</robot>
```

### URDF Components Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                    Robot Definition                     │
│                                                         │
│  ┌─────────────────┐    ┌─────────────────┐            │
│  │     Links       │    │     Joints      │            │
│  │  (Rigid Bodies) │    │ (Constraints &  │            │
│  │                 │    │  Connections)   │            │
│  │  ┌───────────┐  │    │  ┌───────────┐  │            │
│  │  │  Visual   │  │    │  │ Joint Type│  │            │
│  │  │  (Visual) │  │    │  │ (Revolute,│  │            │
│  │  │           │  │    │  │  Fixed, etc│  │            │
│  │  ├───────────┤  │    │  │  )        │  │            │
│  │  │ Collision │  │    │  │           │  │            │
│  │  │  (Physics)│  │    │  │ Limits    │  │            │
│  │  ├───────────┤  │    │  │           │  │            │
│  │  │ Inertial  │  │    │  │ Dynamics  │  │            │
│  │  │  (Mass &  │  │    │  │           │  │            │
│  │  │  Inertia) │  │    │  └───────────┘  │            │
│  │  └───────────┘  │    └─────────────────┘            │
│  └─────────────────┘                                   │
└─────────────────────────────────────────────────────────┘
```

## Links: Defining Robot Bodies

### Link Components

Each link can contain three main components:

```
┌─────────────────┐
│     Link        │
│   (base_link)   │
├─────────────────┤
│  ┌─────────────┐│
│  │   Visual    ││ ←─ For rendering and visualization
│  │             ││
│  │  - Geometry ││
│  │  - Material ││
│  │  - Origin  ││
│  └─────────────┘│
│  ┌─────────────┐│
│  │  Collision  ││ ←─ For physics simulation
│  │             ││
│  │  - Geometry ││
│  │  - Origin  ││
│  └─────────────┘│
│  ┌─────────────┐│
│  │  Inertial   ││ ←─ For dynamics simulation
│  │             ││
│  │  - Mass     ││
│  │  - Inertia  ││
│  │  - Origin  ││
│  └─────────────┘│
└─────────────────┘
```

### Visual Component

The visual component defines how the link appears in visualization tools:

```xml
<link name="link_with_visual">
  <visual>
    <!-- Origin offset from link frame -->
    <origin xyz="0 0 0.1" rpy="0 0 0"/>

    <!-- Geometry defines shape -->
    <geometry>
      <!-- Different geometry types -->
      <box size="0.1 0.2 0.3"/>           <!-- Box: width, depth, height -->
      <!-- <cylinder radius="0.1" length="0.2"/>  Cylinder -->
      <!-- <sphere radius="0.1"/>                   Sphere -->
      <!-- <mesh filename="package://my_pkg/meshes/link.stl"/>  Mesh -->
    </geometry>

    <!-- Material defines color/appearance -->
    <material name="red">
      <color rgba="1 0 0 1"/>  <!-- Red with full opacity -->
    </material>
  </visual>
</link>
```

### Collision Component

The collision component defines the collision geometry for physics simulation:

```xml
<link name="link_with_collision">
  <collision>
    <!-- Often similar to visual but can be simplified -->
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.1 0.2 0.3"/>
    </geometry>
  </collision>
</link>
```

### Inertial Component

The inertial component defines mass properties for dynamics simulation:

```xml
<link name="link_with_inertial">
  <inertial>
    <!-- Mass in kilograms -->
    <mass value="0.5"/>

    <!-- Origin of inertial frame (usually center of mass) -->
    <origin xyz="0 0 0" rpy="0 0 0"/>

    <!-- Inertia matrix (symmetric, only 6 values needed) -->
    <inertia
      ixx="0.01" ixy="0" ixz="0"
      iyy="0.02" iyz="0"
      izz="0.03"/>
  </inertial>
</link>
```

## Joints: Connecting Links

### Joint Types and Properties

```
┌─────────────────────────────────────────────────────────┐
│                    Joint Types                          │
│                                                         │
│  Fixed    │ Revolute │ Continuous │ Prismatic │ Planar │
│  (0 DOF)  │ (1 DOF)  │ (1 DOF)    │ (1 DOF)   │ (3 DOF) │
│           │ Rotational│ Rotational │ Linear    │ XY      │
│           │ Limited   │ Unlimited  │ Movement  │ Movement│
└─────────────────────────────────────────────────────────┘
```

### Joint Definition Example

```xml
<!-- Fixed joint (no movement) -->
<joint name="fixed_joint" type="fixed">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0.1 0 0" rpy="0 0 0"/>
</joint>

<!-- Revolute joint (rotational with limits) -->
<joint name="revolute_joint" type="revolute">
  <parent link="upper_arm"/>
  <child link="lower_arm"/>
  <origin xyz="0 0 0.3" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>  <!-- Rotation axis -->
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
</joint>

<!-- Continuous joint (rotational without limits) -->
<joint name="continuous_joint" type="continuous">
  <parent link="torso"/>
  <child link="head"/>
  <origin xyz="0 0 0.5" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
</joint>

<!-- Prismatic joint (linear movement) -->
<joint name="prismatic_joint" type="prismatic">
  <parent link="base"/>
  <child link="slide"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="0" upper="0.5" effort="50" velocity="0.5"/>
</joint>
```

## Humanoid Robot URDF Example

### Complete Humanoid Robot Structure

```
                    base_link (Root)
                         │
                         ▼
                    torso_link
                   ┌─────┼─────┐
                   │     │     │
              head_link │     │
                   │     │     │
                   ▼     ▼     ▼
              neck_joint │     │
                   │     │     │
                   ▼     ▼     ▼
              neck_link │     │
                   │     │     │
        ┌──────────┼─────┼─────┼──────────┐
        │          │     │     │          │
   left_arm    right_arm │     │     left_leg
    chain       chain    │     │       chain
        │          │     │     │          │
        ▼          ▼     ▼     ▼          ▼
   left_shoulder right_shoulder │     left_hip
        │          │     │     │          │
        ▼          ▼     ▼     ▼          ▼
   left_elbow   right_elbow │     left_knee
        │          │     │     │          │
        ▼          ▼     ▼     ▼          ▼
   left_wrist   right_wrist │     left_ankle
        │          │     │     │          │
        ▼          ▼     ▼     ▼          ▼
   left_hand   right_hand │     left_foot
        │          │     │     │          │
        └──────────┼─────┼─────┼──────────┘
                   │     │     │
              right_leg    │
                   │       │
                   └───────┘
                   chain   chain
```

### Humanoid Robot URDF Fragment

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base and Torso -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.3 0.1"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.3 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso_link"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
  </joint>

  <link name="torso_link">
    <visual>
      <geometry>
        <cylinder radius="0.15" length="0.6"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.15" length="0.6"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.5" ixy="0" ixz="0" iyy="0.5" iyz="0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="torso_to_head" type="revolute">
    <parent link="torso_link"/>
    <child link="head_link"/>
    <origin xyz="0 0 0.35" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="2.0"/>
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
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Left Arm -->
  <joint name="torso_to_left_shoulder" type="revolute">
    <parent link="torso_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0.1 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1.0"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Right Arm (similar structure) -->
  <joint name="torso_to_right_shoulder" type="revolute">
    <parent link="torso_link"/>
    <child link="right_upper_arm"/>
    <origin xyz="0.15 -0.1 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1.0"/>
  </joint>

  <link name="right_upper_arm">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Left Leg -->
  <joint name="torso_to_left_hip" type="revolute">
    <parent link="torso_link"/>
    <child link="left_upper_leg"/>
    <origin xyz="-0.1 0.05 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
  </joint>

  <link name="left_upper_leg">
    <visual>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.02"/>
    </inertial>
  </link>
</robot>
```

## Xacro: URDF Macros and Reusability

### Xacro Introduction

Xacro allows you to define macros and reuse URDF components:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">

  <!-- Define properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="link_radius" value="0.05" />
  <xacro:property name="link_length" value="0.3" />

  <!-- Define a macro for creating arm links -->
  <xacro:macro name="arm_link" params="name xyz parent axis">
    <link name="${name}">
      <visual>
        <geometry>
          <cylinder radius="${link_radius}" length="${link_length}"/>
        </geometry>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${link_radius}" length="${link_length}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="2.0"/>
        <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.01"/>
      </inertial>
    </link>

    <joint name="${parent}_to_${name}" type="revolute">
      <parent link="${parent}"/>
      <child link="${name}"/>
      <origin xyz="${xyz}" rpy="0 0 0"/>
      <axis xyz="${axis}"/>
      <limit lower="-1.57" upper="1.57" effort="50" velocity="1.0"/>
    </joint>
  </xacro:macro>

  <!-- Use the macro to create left and right arms -->
  <xacro:arm_link name="left_upper_arm" xyz="0.15 0.1 0.1" parent="torso_link" axis="0 1 0"/>
  <xacro:arm_link name="right_upper_arm" xyz="0.15 -0.1 0.1" parent="torso_link" axis="0 1 0"/>

</robot>
```

## URDF Tools and Visualization

### URDF Processing Tools

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   URDF File     │───▶│   robot_state_  │───▶│   rviz2         │
│   (.urdf)       │    │   publisher     │    │   Visualization │
└─────────────────┘    │   (Publishes    │    │                 │
         │               │   transforms)   │    │  ┌─────────────┐│
         │               └─────────────────┘    │  │  Robot      ││
         │                        │             │  │  Model      ││
         ▼                        ▼             │  │  Display    ││
┌─────────────────┐    ┌─────────────────┐     │  └─────────────┘│
│   xacro         │───▶│   tf2           │     │  ┌─────────────┐│
│   (Preprocessing)│    │   (Transform    │     │  │  Joint      ││
│                │    │   Library)      │     │  │  Sliders    ││
└─────────────────┘    └─────────────────┘     │  └─────────────┘│
         │                       │              └─────────────────┘
         │                       ▼
         │              ┌─────────────────┐
         └─────────────▶│   gazebo        │
                        │   (Simulation)  │
                        └─────────────────┘
```

### Launching URDF Visualization

```xml
<!-- launch/robot_visualization.launch.py -->
from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Get URDF via xacro
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindPackageShare("my_robot_description"), "urdf", "robot.urdf.xacro"])
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[robot_description],
    )

    # Joint state publisher (for visualization)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
    )

    # Joint state publisher GUI (optional)
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
    )

    # RViz2
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
    )

    return LaunchDescription([
        robot_state_publisher,
        joint_state_publisher,
        joint_state_publisher_gui,
        rviz2
    ])
```

## Best Practices

### URDF Design Guidelines

**Kinematic Chain Structure:**
```
Single Tree Structure:
┌─────────────────┐
│   base_link     │ ← Root of kinematic tree
└─────────┬───────┘
          │
    ┌─────▼─────┐    ┌─────────────────┐
    │  torso    │───▶│   head          │
    │  link     │    │   chain        │
    └───────────┘    └─────────────────┘
          │
    ┌─────▼─────┐    ┌─────────────────┐
    │  arm      │───▶│   left arm      │
    │  chains   │    │   chain         │
    ├───────────┤    └─────────────────┘
    │           │
    └───────────┤    ┌─────────────────┐
              └───▶│   right arm     │
                   │   chain         │
                   └─────────────────┘
```

### Performance Optimization

1. **Simplified Collision Geometry**: Use simple shapes for collision to improve simulation performance
2. **Appropriate Mesh Resolution**: Balance visual detail with performance
3. **Mass Distribution**: Ensure realistic inertial properties for stable simulation
4. **Joint Limits**: Set realistic limits to prevent simulation errors

### Common Mistakes to Avoid

1. **Multiple Base Links**: URDF must have a single root link
2. **Disconnected Components**: All links must be connected through joints
3. **Incorrect Mass Properties**: Ensure realistic mass and inertia values
4. **Overly Complex Geometry**: Use simplified shapes for collision models

## Validation and Debugging

### URDF Validation Tools

```bash
# Check URDF syntax
check_urdf /path/to/robot.urdf

# Print URDF information
urdf_to_graphiz /path/to/robot.urdf

# Visualize kinematic chain
# Output files: robot.pdf (graph) and robot.gv (graphviz)
```

### Common Validation Issues

```
Validation Checklist:
┌─────────────────────────────────────────────────────────┐
│ ✓ Single root link (base_link)                          │
│ ✓ All links connected via joints                        │
│ ✓ Valid XML syntax                                      │
│ ✓ Proper mass and inertia values                        │
│ ✓ Valid joint limits                                    │
│ ✓ No duplicate names                                    │
│ ✓ Correct axis orientations                             │
│ ✓ Reasonable geometry dimensions                        │
└─────────────────────────────────────────────────────────┘
```

## Exercises and Knowledge Checkpoints

### Knowledge Checkpoints

1. What are the three main components of a URDF link?
2. What is the difference between visual and collision components?
3. What are the different joint types available in URDF?
4. What is the purpose of the inertial component?
5. How does Xacro help in creating complex robot models?

### Hands-On Exercise

Create a complete URDF model for a simple humanoid robot with:
1. A base/torso link
2. Head with neck joint
3. Two arms with shoulder, elbow, and wrist joints
4. Two legs with hip, knee, and ankle joints
5. Proper mass, inertia, and visual/collision properties
6. Use Xacro macros to avoid code duplication

Validate your URDF using the `check_urdf` command and visualize it in RViz2.

## Summary

URDF is the standard format for describing robot models in ROS, defining the physical and visual properties of robots through links and joints. For humanoid robots, proper URDF description is crucial for simulation, visualization, and control. Understanding the components of URDF, proper kinematic chain structure, and best practices for modeling enables the creation of accurate and efficient robot descriptions.

## Next Steps

Continue to the next section: [ROS 2 Integration with Humanoid Robots](../nvidia-isaac/introduction-to-isaac-sim.md)