# Unity Robot Visualization in Gazebo Simulation

## Overview
This chapter covers the integration of Unity for advanced 3D visualization of humanoid robots in simulation environments. Unity provides high-quality rendering capabilities that complement Gazebo's physics simulation, creating photorealistic environments for robotics development and testing.

## Learning Objectives
- Understand Unity's role in robotics simulation
- Set up Unity for robot visualization
- Create realistic humanoid robot models in Unity
- Integrate Unity with Gazebo simulation
- Implement advanced rendering techniques
- Optimize performance for real-time visualization

## Prerequisites
- Basic Unity development experience
- Understanding of 3D modeling concepts
- Experience with ROS 2 integration
- Completed "URDF/SDF Simulation Files" chapter
- Unity 2022.3 LTS or later installed

## Table of Contents
1. [Introduction to Unity in Robotics](#introduction-to-unity-in-robotics)
2. [Setting Up Unity for Robotics](#setting-up-unity-for-robotics)
3. [Humanoid Robot Modeling in Unity](#humanoid-robot-modeling-in-unity)
4. [Material and Shader Design](#material-and-shader-design)
5. [Lighting and Environment Design](#lighting-and-environment-design)
6. [ROS 2 Integration with Unity](#ros-2-integration-with-unity)
7. [Gazebo-Unity Bridge](#gazebo-unity-bridge)
8. [Performance Optimization](#performance-optimization)
9. [Best Practices](#best-practices)

## Introduction to Unity in Robotics

### Unity's Role in Robotics Simulation
Unity serves as a high-fidelity visualization layer for robotics simulation, providing:
- Photorealistic rendering capabilities
- Advanced lighting and material systems
- Real-time 3D visualization
- Integration with game engine physics
- VR/AR support for immersive experiences

### Unity vs Gazebo Visualization
While Gazebo provides functional visualization for simulation, Unity offers:
- Higher quality rendering with advanced shaders
- Better lighting models and global illumination
- More sophisticated material properties
- Enhanced post-processing effects
- Better support for complex environments

## Setting Up Unity for Robotics

### Required Packages and Tools
1. **Unity Hub**: For managing Unity versions
2. **Unity 2022.3 LTS**: Recommended for stability
3. **ROS# (ROS Sharp)**: For ROS 2 integration
4. **Universal Render Pipeline (URP)**: For optimized rendering
5. **DOTS (Data-Oriented Technology Stack)**: For performance

### Installation Steps
1. Download and install Unity Hub from unity.com
2. Install Unity 2022.3 LTS through Unity Hub
3. Create a new 3D project
4. Import ROS# package from Unity Asset Store or GitHub
5. Set up Universal Render Pipeline for better performance

### Project Structure
```
UnityRoboticsProject/
├── Assets/
│   ├── Models/           # Robot and environment models
│   ├── Materials/        # Material definitions
│   ├── Shaders/          # Custom shader files
│   ├── Scripts/          # C# scripts for ROS integration
│   ├── Scenes/           # Unity scene files
│   └── Prefabs/          # Reusable robot components
├── Packages/
└── ProjectSettings/
```

## Humanoid Robot Modeling in Unity

### Importing Robot Models
Unity supports various 3D model formats:
- **FBX**: Recommended for complex models with animations
- **OBJ**: Simple format for static geometry
- **DAE**: Collada format for interchange

### Robot Hierarchy Setup
Create a proper hierarchy for humanoid robot kinematics:

```
HumanoidRobot (GameObject)
├── Torso
│   ├── Head
│   ├── LeftShoulder
│   │   ├── LeftArm
│   │   └── LeftHand
│   ├── RightShoulder
│   │   ├── RightArm
│   │   └── RightHand
│   ├── LeftHip
│   │   ├── LeftLeg
│   │   └── LeftFoot
│   └── RightHip
│       ├── RightLeg
│       └── RightFoot
```

### Joint Configuration
Configure joints to match robot kinematics:

```csharp
// Example: Configuring a revolute joint
public class RobotJoint : MonoBehaviour
{
    public float minAngle = -90f;
    public float maxAngle = 90f;
    public float currentAngle = 0f;

    public void SetJointAngle(float angle)
    {
        currentAngle = Mathf.Clamp(angle, minAngle, maxAngle);
        transform.localRotation = Quaternion.Euler(0, currentAngle, 0);
    }
}
```

### Animation and Control
Use Unity's Animation system for complex humanoid movements:
1. **Animator Controller**: For state-based animations
2. **Animation Clips**: For specific movement sequences
3. **Inverse Kinematics**: For realistic limb positioning

## Material and Shader Design

### Physically-Based Materials
Create realistic materials using PBR (Physically Based Rendering):

```hlsl
Shader "Robotics/PBRRobotMaterial"
{
    Properties
    {
        _Color ("Color", Color) = (1,1,1,1)
        _MainTex ("Albedo", 2D) = "white" {}
        _Metallic ("Metallic", Range(0,1)) = 0.0
        _Smoothness ("Smoothness", Range(0,1)) = 0.5
        _BumpMap ("Normal Map", 2D) = "bump" {}
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 200

        CGPROGRAM
        #pragma surface surf Standard fullforwardshadows
        #pragma target 3.0

        struct Input
        {
            float2 uv_MainTex;
        };

        sampler2D _MainTex;
        fixed4 _Color;
        half _Metallic;
        half _Smoothness;

        void surf (Input IN, inout SurfaceOutputStandard o)
        {
            fixed4 c = tex2D (_MainTex, IN.uv_MainTex) * _Color;
            o.Albedo = c.rgb;
            o.Metallic = _Metallic;
            o.Smoothness = _Smoothness;
            o.Alpha = c.a;
        }
        ENDCG
    }
}
```

### Sensor Visualization Materials
Create special materials for sensor visualization:
- **Camera FOV**: Transparent cones showing field of view
- **LiDAR rays**: Thin lines showing sensor beams
- **Collision detection**: Highlight materials for debugging

## Lighting and Environment Design

### Realistic Lighting Setup
Create realistic lighting for humanoid robot environments:

```csharp
public class RoboticsLightingSetup : MonoBehaviour
{
    public Light mainLight;
    public Light fillLight;
    public Light rimLight;

    void Start()
    {
        // Configure main light (key light)
        mainLight.type = LightType.Directional;
        mainLight.color = Color.white;
        mainLight.intensity = 1.5f;
        mainLight.shadows = LightShadows.Soft;

        // Configure fill light
        fillLight.type = LightType.Directional;
        fillLight.color = Color.gray;
        fillLight.intensity = 0.5f;

        // Configure rim light for depth
        rimLight.type = LightType.Directional;
        rimLight.color = Color.white;
        rimLight.intensity = 0.3f;
    }
}
```

### Environment Design for Humanoid Robots
Design environments suitable for humanoid robot simulation:
- **Corridors**: Appropriate width for humanoid navigation
- **Doorways**: Standard human-sized passages
- **Furniture**: Appropriately sized for human interaction
- **Stairs**: With proper riser and tread dimensions

### Post-Processing Effects
Implement post-processing for realistic camera simulation:
- **Depth of Field**: Simulate camera focus
- **Motion Blur**: For realistic movement perception
- **Color Grading**: Match real camera characteristics
- **Lens Distortion**: Simulate real lens properties

## ROS 2 Integration with Unity

### ROS# Setup
Configure ROS# for Unity-ROS 2 communication:

1. Install ROS# package in Unity
2. Set up ROS connection in Unity scene
3. Configure message types for robot data
4. Implement publisher/subscriber patterns

### Robot State Visualization
Create scripts to visualize robot state from ROS topics:

```csharp
using ROS2;
using UnityEngine;

public class RobotStateVisualizer : MonoBehaviour
{
    private ROS2UnityComponent ros2Unity;
    private ISubscription<sensor_msgs.msg.JointState> jointStateSub;

    void Start()
    {
        ros2Unity = GetComponent<ROS2UnityComponent>();
        ros2Unity.Initialize();

        jointStateSub = ros2Unity.node.CreateSubscription<sensor_msgs.msg.JointState>(
            "/joint_states", JointStateCallback);
    }

    void JointStateCallback(sensor_msgs.msg.JointState msg)
    {
        // Update robot joint positions based on ROS message
        for (int i = 0; i < msg.name.Count; i++)
        {
            Transform joint = FindJointByName(msg.name[i]);
            if (joint != null)
            {
                joint.localRotation = Quaternion.Euler(0, msg.position[i], 0);
            }
        }
    }

    Transform FindJointByName(string name)
    {
        Transform[] joints = GetComponentsInChildren<Transform>();
        foreach (Transform joint in joints)
        {
            if (joint.name == name)
                return joint;
        }
        return null;
    }
}
```

### Sensor Data Visualization
Visualize sensor data from ROS topics in Unity:
- **Camera images**: Display on UI elements
- **LiDAR data**: Visualize as point clouds
- **IMU data**: Show orientation and acceleration
- **Force/torque sensors**: Visualize as arrows

## Gazebo-Unity Bridge

### Architecture Overview
The bridge architecture typically involves:
1. **Gazebo**: Physics simulation and sensor simulation
2. **ROS 2**: Communication middleware
3. **Unity**: High-fidelity visualization
4. **Bridge node**: Translates between systems

### Bridge Implementation
Create a bridge node that synchronizes state between Gazebo and Unity:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
import socket
import json

class GazeboUnityBridge(Node):
    def __init__(self):
        super().__init__('gazebo_unity_bridge')

        # Subscribe to Gazebo state
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        # Setup Unity connection
        self.unity_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.unity_socket.connect(('localhost', 5555))

    def joint_state_callback(self, msg):
        # Convert ROS message to Unity format
        unity_data = {
            'joint_names': msg.name,
            'joint_positions': list(msg.position),
            'joint_velocities': list(msg.velocity),
            'timestamp': self.get_clock().now().nanoseconds
        }

        # Send to Unity
        self.unity_socket.send(json.dumps(unity_data).encode())

def main(args=None):
    rclpy.init(args=args)
    bridge = GazeboUnityBridge()
    rclpy.spin(bridge)
    bridge.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization

### Rendering Optimization
Optimize Unity rendering for real-time performance:

1. **LOD (Level of Detail)**: Use different model complexities
2. **Occlusion Culling**: Hide objects not in view
3. **Texture Atlasing**: Combine multiple textures
4. **Shader Optimization**: Use efficient shaders for mobile/desktop

### Robot Model Optimization
For humanoid robots with many joints:
- Use skinned meshes for organic parts
- Use static meshes for rigid components
- Limit the number of bones in skinned meshes
- Use instancing for multiple similar robots

### Multi-Threading
Implement multi-threading for simulation updates:
```csharp
using System.Threading.Tasks;

public class MultiThreadedRobotUpdater : MonoBehaviour
{
    async void UpdateRobotStates()
    {
        await Task.Run(() => {
            // Update robot states in background thread
            UpdateJointPositions();
            ProcessSensorData();
        });
    }
}
```

## Best Practices

1. **Modular Components**: Create reusable robot components as prefabs
2. **Consistent Naming**: Use consistent naming conventions for joints and links
3. **Performance Testing**: Regularly test performance on target hardware
4. **Version Control**: Track 3D assets with appropriate version control
5. **Documentation**: Document complex shader and script functionality
6. **Quality Settings**: Configure Unity quality settings for target platform
7. **Testing Pipeline**: Implement automated testing for visualization changes

## Exercises

1. Set up a Unity project with ROS# integration
2. Create a simple humanoid robot model with basic joints
3. Implement a material system for robot visualization
4. Configure lighting for a humanoid robot environment
5. Create a basic bridge between Gazebo and Unity
6. Optimize a robot model for real-time visualization

## NVIDIA Isaac Sim Integration

### Overview of Isaac Sim
NVIDIA Isaac Sim is a robotics simulator built on NVIDIA Omniverse, designed for developing and testing AI-based robotics applications. It provides high-fidelity physics simulation, photorealistic rendering, and integration with NVIDIA's AI and robotics platforms.

### Key Features of Isaac Sim
- **Photorealistic Rendering**: RTX-accelerated rendering for realistic sensor simulation
- **High-Fidelity Physics**: PhysX-based physics engine for accurate simulation
- **AI Integration**: Direct integration with NVIDIA AI frameworks
- **Omniverse Platform**: Real-time collaboration and extensibility
- **Synthetic Data Generation**: Tools for generating training data for AI models

### Installing Isaac Sim
Isaac Sim is part of the Isaac ROS ecosystem and can be installed in several ways:

#### Docker Installation (Recommended)
```bash
# Pull the Isaac Sim Docker image
docker pull nvcr.io/nvidia/isaac-sim:4.2.0

# Run Isaac Sim in Docker
docker run --gpus all -it --rm \
  --network=host \
  --env "ACCEPT_EULA=Y" \
  --env "INSTALL_PATH=/isaac-sim" \
  --volume $(pwd):/workspace/isaac-sim \
  --volume ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  --volume ~/docker/isaac-sim/cache/ov:/root/.ov/cache:rw \
  --volume ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  --volume ~/docker/isaac-sim/config:/root/.nvidia-omniverse/config:rw \
  --volume ~/docker/isaac-sim/data:/isaac-sim/assets:rw \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --env "DISPLAY=$DISPLAY" \
  --privileged \
  nvcr.io/nvidia/isaac-sim:4.2.0
```

#### Standalone Installation
1. Download Isaac Sim from NVIDIA Developer website
2. Extract the package to your desired location
3. Run the setup script: `./isaac-sim/setup.sh`
4. Launch Isaac Sim: `./isaac-sim/runheadless.sh` or `./isaac-sim/run.sh`

### Isaac Sim vs Gazebo
| Feature | Isaac Sim | Gazebo |
|---------|-----------|---------|
| Rendering | RTX-photorealistic | Basic OpenGL |
| Physics | PhysX (high-fidelity) | ODE, Bullet, DART |
| AI Integration | Native NVIDIA AI tools | Plugin-based |
| Performance | GPU-accelerated | CPU-based primarily |
| Sensor Simulation | Advanced (LiDAR, Cameras, etc.) | Standard sensor models |

### Creating Humanoid Robots in Isaac Sim
Isaac Sim uses USD (Universal Scene Description) format for scene and robot descriptions. Here's how to create a humanoid robot:

#### Basic Robot USD Structure
```
humanoid_robot.usd
├── Robot
│   ├── Torso
│   ├── Head
│   ├── LeftArm
│   │   ├── LeftShoulder
│   │   ├── LeftElbow
│   │   └── LeftHand
│   ├── RightArm
│   │   ├── RightShoulder
│   │   ├── RightElbow
│   │   └── RightHand
│   ├── LeftLeg
│   │   ├── LeftHip
│   │   ├── LeftKnee
│   │   └── LeftFoot
│   └── RightLeg
│       ├── RightHip
│       ├── RightKnee
│       └── RightFoot
```

#### Example Robot Definition
```python
# Python example using Omniverse Kit
import omni
from pxr import Usd, UsdGeom, Gf, Sdf

def create_humanoid_robot(stage, path):
    """Create a basic humanoid robot in USD"""

    # Create the robot prim
    robot_prim = UsdGeom.Xform.Define(stage, path)

    # Create torso
    torso_path = path + "/torso"
    torso = UsdGeom.Cube.Define(stage, torso_path)
    torso.GetSizeAttr().Set(0.3)
    torso.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.5))

    # Create head
    head_path = path + "/head"
    head = UsdGeom.Sphere.Define(stage, head_path)
    head.GetRadiusAttr().Set(0.1)
    head.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.8))

    # Create limbs with articulation components
    create_arm(stage, path + "/left_arm", is_left=True)
    create_arm(stage, path + "/right_arm", is_left=False)
    create_leg(stage, path + "/left_leg", is_left=True)
    create_leg(stage, path + "/right_leg", is_left=False)

    return robot_prim

def create_arm(stage, path, is_left=True):
    """Create an articulated arm"""
    side_mult = -1 if is_left else 1

    # Shoulder
    shoulder_path = path + "/shoulder"
    shoulder = UsdGeom.Cylinder.Define(stage, shoulder_path)
    shoulder.GetRadiusAttr().Set(0.05)
    shoulder.GetHeightAttr().Set(0.1)
    shoulder.AddTranslateOp().Set(Gf.Vec3d(side_mult * 0.2, 0, 0.6))

    # Upper arm
    upper_arm_path = path + "/upper_arm"
    upper_arm = UsdGeom.Cylinder.Define(stage, upper_arm_path)
    upper_arm.GetRadiusAttr().Set(0.04)
    upper_arm.GetHeightAttr().Set(0.3)
    upper_arm.AddTranslateOp().Set(Gf.Vec3d(side_mult * 0.35, 0, 0.6))

    # Lower arm
    lower_arm_path = path + "/lower_arm"
    lower_arm = UsdGeom.Cylinder.Define(stage, lower_arm_path)
    lower_arm.GetRadiusAttr().Set(0.035)
    lower_arm.GetHeightAttr().Set(0.3)
    lower_arm.AddTranslateOp().Set(Gf.Vec3d(side_mult * 0.55, 0, 0.6))

# Articulation components would be added separately
```

### Isaac Sim for Humanoid Robotics
Isaac Sim provides several advantages for humanoid robotics:

#### Advanced Sensor Simulation
- **RGB Cameras**: With realistic lens distortion and noise models
- **Depth Cameras**: Accurate depth perception simulation
- **LiDAR**: High-fidelity LiDAR simulation with configurable parameters
- **IMU Simulation**: Realistic inertial measurement unit simulation
- **Force/Torque Sensors**: Accurate contact force simulation

#### Physics Simulation
- **PhysX Engine**: High-fidelity physics simulation
- **Contact Materials**: Realistic friction and contact properties
- **Soft Body Dynamics**: Simulation of flexible materials
- **Fluid Simulation**: For environmental interactions

#### AI Training Integration
- **Synthetic Data Generation**: Tools for generating labeled training data
- **Domain Randomization**: Techniques to improve model generalization
- **Reinforcement Learning**: Integration with RL training frameworks
- **Perception Training**: Tools for training perception models

### Isaac Sim ROS 2 Bridge
Isaac Sim provides excellent integration with ROS 2 through the Isaac ROS packages:

#### Installation
```bash
# Install Isaac ROS packages
sudo apt update
sudo apt install ros-humble-isaac-ros-gem ros-humble-isaac-ros-visual- slam ros-humble-isaac-ros-segmentation ros-humble-isaac-ros-audio
```

#### ROS 2 Integration Example
```python
#!/usr/bin/env python3
# Example of ROS 2 integration with Isaac Sim
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

class IsaacSimController(Node):
    def __init__(self):
        super().__init__('isaac_sim_controller')

        # Subscribe to joint commands
        self.joint_sub = self.create_subscription(
            JointState, '/isaac_sim/joint_commands',
            self.joint_command_callback, 10)

        # Subscribe to base velocity commands
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # Publish sensor data
        self.joint_pub = self.create_publisher(
            JointState, '/joint_states', 10)

    def joint_command_callback(self, msg):
        # Send joint commands to Isaac Sim
        # This would interface with Isaac Sim's control API
        pass

    def cmd_vel_callback(self, msg):
        # Send velocity commands to simulated robot
        pass

def main(args=None):
    rclpy.init(args=args)
    controller = IsaacSimController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Isaac ROS Perception Packages
Isaac Sim includes specialized ROS packages for perception tasks:

#### Isaac ROS Image Pipeline
- **Image Proc**: Image processing and rectification
- **Stereo Image Proc**: Stereo vision processing
- **Image Transport**: Efficient image transport

#### Isaac ROS Navigation
- **Visual SLAM**: Visual Simultaneous Localization and Mapping
- **Path Planning**: GPU-accelerated path planning
- **Collision Avoidance**: Real-time collision avoidance

### Synthetic Data Generation
One of Isaac Sim's key strengths is synthetic data generation:

#### Domain Randomization
```python
# Example: Randomizing environment for synthetic data
import omni
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage
import random

def randomize_environment():
    """Randomize environment properties for synthetic data"""

    # Randomize lighting
    light_intensity = random.uniform(500, 1500)
    light_color = [random.uniform(0.8, 1.2), random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)]

    # Randomize textures
    texture_variations = [
        "textures/metal_1.usd",
        "textures/concrete_1.usd",
        "textures/wood_1.usd"
    ]
    random_texture = random.choice(texture_variations)

    # Randomize object positions
    for i in range(10):
        x_pos = random.uniform(-5, 5)
        y_pos = random.uniform(-5, 5)
        z_pos = random.uniform(0, 2)
        # Apply random position to objects
```

### Performance Considerations
When using Isaac Sim for humanoid robotics:

#### Hardware Requirements
- **GPU**: NVIDIA RTX 3080 or better (RTX 4090 recommended)
- **VRAM**: 12GB+ for complex scenes
- **CPU**: Multi-core processor (8+ cores)
- **RAM**: 32GB+ for large environments
- **Storage**: SSD with 100GB+ free space

#### Optimization Techniques
- **Level of Detail (LOD)**: Use simplified models when far from camera
- **Occlusion Culling**: Hide objects not visible to sensors
- **Simulation Frequency**: Balance accuracy with performance
- **Batch Processing**: Generate multiple synthetic data samples in parallel

### Troubleshooting Common Issues

#### Rendering Issues
- Ensure NVIDIA GPU drivers are up to date
- Check that RTX features are enabled in BIOS
- Verify CUDA toolkit is properly installed

#### Performance Issues
- Reduce scene complexity for real-time simulation
- Use lower resolution textures during development
- Limit the number of active sensors

#### Integration Problems
- Verify ROS 2 environment is properly sourced
- Check network connectivity between Isaac Sim and ROS nodes
- Ensure correct message types and topics

## Unity-Isaac Sim Integration

### Architecture Overview
For humanoid robotics applications, you might want to integrate Unity visualization with Isaac Sim's physics and perception capabilities:

1. **Isaac Sim**: Handles physics simulation and sensor data generation
2. **ROS 2**: Provides communication middleware
3. **Unity**: Provides high-quality visualization
4. **Custom Bridge**: Synchronizes state between systems

### Best Practices

1. **Start Simple**: Begin with basic robot models before adding complexity
2. **Validate Results**: Compare simulation results with real-world data
3. **Documentation**: Maintain clear documentation of simulation parameters
4. **Version Control**: Track changes to USD files and simulation configurations
5. **Testing**: Regularly test simulation-to-reality transfer

## Exercises

1. Set up a Unity project with ROS# integration and establish communication with a ROS 2 system
2. Create a simple humanoid robot model in Unity with articulated joints
3. Implement a material system with realistic PBR shaders for robot components
4. Configure realistic lighting for a humanoid robot environment in Unity
5. Create a basic bridge node that synchronizes robot state between Gazebo and Unity
6. Implement a sensor visualization system that shows camera FOV and LiDAR beams in Unity

## Next Steps

After completing this chapter, proceed to learn about synthetic data generation techniques to understand how to create training datasets for AI models using simulation environments.