# Kinematics and Dynamics

## Overview

Kinematics and dynamics form the mathematical foundation for understanding and controlling humanoid robot motion. Kinematics deals with the geometric relationships between robot joints and end-effectors without considering forces, while dynamics encompasses the forces and torques that cause motion. This chapter provides a comprehensive treatment of these essential concepts for humanoid robotics, covering both theoretical foundations and practical implementation approaches.

Understanding kinematics and dynamics is crucial for developing stable, efficient, and human-like movements in humanoid robots. The complex multi-link structure of humanoid robots, combined with the need for balance and coordination, makes these topics particularly important for successful humanoid robot development.

## Learning Objectives

By the end of this chapter, you should be able to:

- Understand forward and inverse kinematics for humanoid robots
- Apply dynamics principles to analyze and control robot motion
- Implement kinematic and dynamic models for humanoid robots
- Analyze stability and balance in humanoid systems
- Design control strategies based on kinematic and dynamic models
- Apply optimization techniques for motion planning
- Understand the relationship between human biomechanics and robot design

## Table of Contents

1. [Introduction to Robot Kinematics](#introduction-to-robot-kinematics)
2. [Forward Kinematics](#forward-kinematics)
3. [Inverse Kinematics](#inverse-kinematics)
4. [Differential Kinematics](#differential-kinematics)
5. [Robot Dynamics](#robot-dynamics)
6. [Humanoid-Specific Kinematics](#humanoid-specific-kinematics)
7. [Balance and Stability](#balance-and-stability)
8. [Implementation Considerations](#implementation-considerations)
9. [Exercises](#exercises)

## Introduction to Robot Kinematics

### What is Kinematics?

Robot kinematics is the study of motion without considering the forces that cause it. In robotics, kinematics deals with the relationship between joint angles and the position and orientation of robot end-effectors (hands, feet, etc.).

For humanoid robots, kinematics becomes particularly complex due to the large number of degrees of freedom (DOF) and the need to coordinate multiple limbs simultaneously while maintaining balance.

### Coordinate Systems and Representations

#### Denavit-Hartenberg (DH) Parameters
The DH convention provides a systematic way to define coordinate frames for robot links:

```python
import numpy as np

def dh_transform(a, alpha, d, theta):
    """
    Calculate Denavit-Hartenberg transformation matrix
    a: link length
    alpha: link twist
    d: link offset
    theta: joint angle
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    T = np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])
    return T

def forward_kinematics_DH(dh_params, joint_angles):
    """
    Calculate forward kinematics using DH parameters
    dh_params: List of [a, alpha, d, joint_type] for each joint
    joint_angles: List of joint angles (for revolute joints)
    """
    T_total = np.eye(4)

    for i, (a, alpha, d, joint_type) in enumerate(dh_params):
        if joint_type == 'revolute':
            theta = joint_angles[i]
        else:  # prismatic
            theta = 0  # or use joint_angles[i] for prismatic joint

        T_link = dh_transform(a, alpha, d, theta)
        T_total = T_total @ T_link

    return T_total
```

#### Rotation Matrices and Quaternions
Rotation representations are crucial for describing orientations:

```python
def rotation_matrix_x(angle):
    """Rotation matrix around X axis"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

def rotation_matrix_y(angle):
    """Rotation matrix around Y axis"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

def rotation_matrix_z(angle):
    """Rotation matrix around Z axis"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles to quaternion"""
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix"""
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z

    if Nq < 1e-8:
        return np.eye(3)

    s = 2.0 / Nq
    X = x*s; Y = y*s; Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z

    R = np.array([
        [1.0-(yY+zZ), xY-wZ, xZ+wY],
        [xY+wZ, 1.0-(xX+zZ), yZ-wX],
        [xZ-wY, yZ+wX, 1.0-(xX+yY)]
    ])
    return R
```

### Degrees of Freedom and Configuration Space

Humanoid robots typically have 20-30+ degrees of freedom, with each limb contributing multiple joints:

- **Leg**: Hip (3 DOF), Knee (1 DOF), Ankle (2 DOF) = 6 DOF per leg
- **Arm**: Shoulder (3 DOF), Elbow (1 DOF), Wrist (3 DOF) = 7 DOF per arm
- **Torso**: Multiple joints for flexibility
- **Head**: Neck joints for gaze control

## Forward Kinematics

### Chain Composition

Forward kinematics calculates the end-effector position and orientation given joint angles:

```python
class ForwardKinematics:
    def __init__(self, dh_params):
        self.dh_params = dh_params  # List of [a, alpha, d, joint_type] for each joint

    def calculate_link_transform(self, link_idx, joint_angle):
        """Calculate transformation matrix for a single link"""
        a, alpha, d, joint_type = self.dh_params[link_idx]

        if joint_type == 'revolute':
            theta = joint_angle
        else:  # prismatic
            theta = 0

        return dh_transform(a, alpha, d, theta)

    def calculate_end_effector_pose(self, joint_angles):
        """Calculate end-effector pose given joint angles"""
        T_total = np.eye(4)

        for i, angle in enumerate(joint_angles):
            T_link = self.calculate_link_transform(i, angle)
            T_total = T_total @ T_link

        return T_total  # 4x4 transformation matrix

    def calculate_all_link_poses(self, joint_angles):
        """Calculate pose of all links in the chain"""
        poses = [np.eye(4)]  # Base pose

        T_current = np.eye(4)
        for i, angle in enumerate(joint_angles):
            T_link = self.calculate_link_transform(i, angle)
            T_current = T_current @ T_link
            poses.append(T_current.copy())

        return poses
```

### Humanoid Robot Kinematic Chains

Humanoid robots have multiple kinematic chains (arms, legs) that must be coordinated:

```python
class HumanoidKinematics:
    def __init__(self):
        # Define kinematic chains for different limbs
        self.right_arm_chain = self.define_arm_chain()
        self.left_arm_chain = self.define_arm_chain()
        self.right_leg_chain = self.define_leg_chain()
        self.left_leg_chain = self.define_leg_chain()

    def define_arm_chain(self):
        """Define DH parameters for arm chain"""
        # Simplified arm with 7 DOF
        # [a, alpha, d, joint_type]
        dh_params = [
            [0, np.pi/2, 0, 'revolute'],      # Shoulder Yaw
            [0, -np.pi/2, 0, 'revolute'],     # Shoulder Pitch
            [0, np.pi/2, 0, 'revolute'],      # Shoulder Roll
            [0.1, 0, 0, 'revolute'],          # Elbow
            [0, np.pi/2, 0, 'revolute'],      # Wrist Yaw
            [0, -np.pi/2, 0, 'revolute'],     # Wrist Pitch
            [0, 0, 0.1, 'revolute']           # Wrist Roll
        ]
        return ForwardKinematics(dh_params)

    def define_leg_chain(self):
        """Define DH parameters for leg chain"""
        # Simplified leg with 6 DOF
        dh_params = [
            [0, np.pi/2, 0, 'revolute'],      # Hip Yaw
            [0, -np.pi/2, 0, 'revolute'],     # Hip Roll
            [0, 0, 0, 'revolute'],            # Hip Pitch
            [0, 0, -0.4, 'revolute'],         # Knee
            [0, 0, -0.4, 'revolute'],         # Ankle Pitch
            [0, 0, 0, 'revolute']             # Ankle Roll
        ]
        return ForwardKinematics(dh_params)

    def calculate_arm_pose(self, arm, joint_angles):
        """Calculate pose for specified arm"""
        if arm == 'right':
            return self.right_arm_chain.calculate_end_effector_pose(joint_angles)
        elif arm == 'left':
            return self.left_arm_chain.calculate_end_effector_pose(joint_angles)
        else:
            raise ValueError("Arm must be 'right' or 'left'")

    def calculate_leg_pose(self, leg, joint_angles):
        """Calculate pose for specified leg"""
        if leg == 'right':
            return self.right_leg_chain.calculate_end_effector_pose(joint_angles)
        elif leg == 'left':
            return self.left_leg_chain.calculate_end_effector_pose(joint_angles)
        else:
            raise ValueError("Leg must be 'right' or 'left'")
```

### Jacobian Matrices

The Jacobian relates joint velocities to end-effector velocities:

```python
def calculate_jacobian(robot, joint_angles, link_idx):
    """
    Calculate geometric Jacobian for a robot
    """
    n = len(joint_angles)
    J = np.zeros((6, n))  # 6xN Jacobian (3 pos + 3 rot)

    # Calculate all link poses
    link_poses = robot.calculate_all_link_poses(joint_angles)
    end_effector_pose = link_poses[-1]

    # Calculate base position and orientation
    end_pos = end_effector_pose[:3, 3]

    for i in range(n):
        # Get z-axis of joint i (in world frame)
        z_i = link_poses[i][:3, 2]  # Third column is z-axis

        # Get position of joint i (in world frame)
        p_i = link_poses[i][:3, 3]

        # Calculate Jacobian column
        # Position part: z_i × (end_pos - p_i)
        pos_part = np.cross(z_i, end_pos - p_i)

        # Orientation part: z_i
        J[:3, i] = pos_part
        J[3:, i] = z_i

    return J

def jacobian_pseudoinverse(J, damping=0.01):
    """Calculate damped pseudoinverse of Jacobian"""
    J_pinv = np.linalg.pinv(J, rcond=damping)
    return J_pinv

def inverse_velocity_kinematics(J, end_effector_velocity, damping=0.01):
    """Calculate joint velocities from end-effector velocity"""
    J_pinv = jacobian_pseudoinverse(J, damping)
    joint_velocities = J_pinv @ end_effector_velocity
    return joint_velocities
```

## Inverse Kinematics

### Analytical vs Numerical Methods

Inverse kinematics (IK) is the process of finding joint angles that achieve a desired end-effector pose. For humanoid robots, this is typically solved using numerical methods due to the complexity.

#### Analytical Solutions
For simple chains, analytical solutions may exist:

```python
def solve_2dof_arm_ik(x, y, l1, l2):
    """
    Analytical solution for 2-DOF planar arm
    x, y: desired end-effector position
    l1, l2: link lengths
    """
    # Distance from origin to desired position
    r = np.sqrt(x**2 + y**2)

    # Check if position is reachable
    if r > l1 + l2:
        # Position too far
        scale = (l1 + l2) / r
        x *= scale
        y *= scale
        r = l1 + l2
    elif r < abs(l1 - l2):
        # Position too close
        scale = abs(l1 - l2) / r
        x *= scale
        y *= scale
        r = abs(l1 - l2)

    # Calculate second joint angle
    cos_theta2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_theta2 = np.clip(cos_theta2, -1, 1)  # Clamp to valid range
    theta2 = np.arccos(cos_theta2)

    # Calculate first joint angle
    k1 = l1 + l2 * np.cos(theta2)
    k2 = l2 * np.sin(theta2)

    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

    return theta1, theta2
```

#### Numerical Methods

For complex humanoid robots, numerical methods are typically used:

```python
def inverse_kinematics_numerical(
    robot,
    target_pose,
    initial_joints,
    max_iterations=100,
    tolerance=1e-4
):
    """
    Solve inverse kinematics using numerical method (Jacobian transpose)
    """
    joints = np.array(initial_joints, dtype=float)

    for i in range(max_iterations):
        # Calculate current end-effector pose
        current_pose = robot.calculate_end_effector_pose(joints)

        # Calculate error
        pos_error = target_pose[:3, 3] - current_pose[:3, 3]
        rot_error = rotation_error(target_pose, current_pose)

        error = np.concatenate([pos_error, rot_error])

        # Check convergence
        if np.linalg.norm(error) < tolerance:
            break

        # Calculate Jacobian
        J = calculate_jacobian(robot, joints, -1)

        # Update joint angles using Jacobian transpose
        # Note: For better convergence, pseudoinverse is often better
        delta_joints = 0.1 * J.T @ error  # Learning rate = 0.1
        joints += delta_joints

    return joints

def rotation_error(R1, R2):
    """Calculate rotation error between two rotation matrices"""
    # Convert to axis-angle representation
    R_error = R1[:3, :3] @ R2[:3, :3].T
    angle = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))

    # Calculate axis
    if angle < 1e-6:
        return np.zeros(3)

    axis = np.array([
        R_error[2, 1] - R_error[1, 2],
        R_error[0, 2] - R_error[2, 0],
        R_error[1, 0] - R_error[0, 1]
    ]) / (2 * np.sin(angle))

    return angle * axis
```

### Optimization-Based IK

For humanoid robots, optimization-based approaches are often more suitable:

```python
from scipy.optimize import minimize

def ik_optimization_objective(joint_angles, robot, target_pose, weights=None):
    """
    Objective function for IK optimization
    """
    if weights is None:
        weights = np.ones(6)  # Equal weight for position and orientation

    current_pose = robot.calculate_end_effector_pose(joint_angles)

    # Position error
    pos_error = target_pose[:3, 3] - current_pose[:3, 3]

    # Orientation error (using rotation vector representation)
    R_error = target_pose[:3, :3] @ current_pose[:3, :3].T
    rot_vec = rotation_matrix_to_axis_angle(R_error)

    # Weighted error
    error = np.concatenate([pos_error, rot_vec])
    weighted_error = error * weights

    return np.sum(weighted_error**2)

def rotation_matrix_to_axis_angle(R):
    """Convert rotation matrix to axis-angle representation"""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

    if angle < 1e-6:
        return np.zeros(3)

    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2 * np.sin(angle))

    return angle * axis

def solve_ik_optimization(robot, target_pose, initial_joints, joint_limits=None):
    """
    Solve IK using optimization
    """
    def objective(joints):
        return ik_optimization_objective(joints, robot, target_pose)

    # Set up constraints
    constraints = []
    if joint_limits:
        for i, (min_val, max_val) in enumerate(joint_limits):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, min_val=min_val: x[i] - min_val})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, max_val=max_val: max_val - x[i]})

    # Solve optimization problem
    result = minimize(
        objective,
        initial_joints,
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': 1000}
    )

    return result.x if result.success else initial_joints
```

## Differential Kinematics

### Velocity Kinematics

Differential kinematics relates joint velocities to end-effector velocities through the Jacobian matrix:

```python
class DifferentialKinematics:
    def __init__(self, robot):
        self.robot = robot

    def calculate_velocity(self, joint_angles, joint_velocities):
        """Calculate end-effector velocity from joint velocities"""
        J = calculate_jacobian(self.robot, joint_angles, -1)
        end_vel = J @ joint_velocities
        return end_vel  # [vx, vy, vz, wx, wy, wz]

    def calculate_acceleration(self, joint_angles, joint_velocities, joint_accelerations):
        """Calculate end-effector acceleration"""
        J = calculate_jacobian(self.robot, joint_angles, -1)
        end_acc = J @ joint_accelerations

        # Add Coriolis and centrifugal terms (simplified)
        # In practice, this would require the derivative of the Jacobian
        return end_acc

    def inverse_velocity(self, joint_angles, end_velocity, damping=0.01):
        """Calculate joint velocities from end-effector velocity"""
        J = calculate_jacobian(self.robot, joint_angles, -1)
        J_pinv = jacobian_pseudoinverse(J, damping)
        joint_velocities = J_pinv @ end_velocity
        return joint_velocities
```

### Jacobian Derivatives

For more accurate dynamic modeling, Jacobian derivatives may be needed:

```python
def calculate_jacobian_derivative(robot, joint_angles, joint_velocities):
    """
    Calculate the time derivative of the Jacobian
    This is needed for acceleration-level kinematics
    """
    n = len(joint_angles)
    J_dot = np.zeros((6, n))

    # Get all link poses and velocities
    link_poses = robot.calculate_all_link_poses(joint_angles)

    # Calculate end-effector velocity and acceleration
    J = calculate_jacobian(robot, joint_angles, -1)
    end_vel = J @ joint_velocities

    # Calculate J_dot using the relationship:
    # J_dot * q_dot = end_acc - J * q_ddot
    # This is a simplified calculation

    for i in range(n):
        # For each column of J_dot, we need to consider
        # how the Jacobian changes with joint motion
        pass  # Implementation would depend on specific robot structure

    return J_dot
```

## Robot Dynamics

### Rigid Body Dynamics

Robot dynamics describes the relationship between forces/torques and motion:

```python
def rigid_body_inertia(mass, com, inertia_tensor):
    """
    Create 6x6 inertia matrix for a rigid body
    mass: mass of the body
    com: center of mass (3x1 vector)
    inertia_tensor: 3x3 inertia tensor about COM
    """
    I_matrix = np.zeros((6, 6))

    # Upper left: mass matrix
    I_matrix[:3, :3] = mass * np.eye(3)

    # Upper right: cross product matrix for COM
    S_com = np.array([
        [0, -com[2], com[1]],
        [com[2], 0, -com[0]],
        [-com[1], com[0], 0]
    ])
    I_matrix[:3, 3:] = mass * S_com

    # Lower left: negative of upper right
    I_matrix[3:, :3] = -mass * S_com

    # Lower right: inertia tensor
    I_matrix[3:, 3:] = inertia_tensor

    return I_matrix

def transform_inertia(I_body, transform):
    """
    Transform inertia matrix from one frame to another
    """
    R = transform[:3, :3]  # Rotation part
    p = transform[:3, 3]   # Translation part

    # Translation matrix
    S_p = np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0]
    ])

    # Transformation matrix
    T = np.eye(6)
    T[:3, :3] = R
    T[3:, 3:] = R
    T[:3, 3:] = S_p @ R

    # Transform inertia
    I_transformed = T @ I_body @ T.T
    return I_transformed
```

### Euler-Lagrange Formulation

The Euler-Lagrange formulation provides the dynamic equations of motion:

```python
class RobotDynamics:
    def __init__(self, robot_model):
        self.model = robot_model
        self.n = len(robot_model.joint_names)  # Number of joints

    def mass_matrix(self, joint_angles):
        """
        Calculate mass matrix H(q) using composite rigid body algorithm
        """
        H = np.zeros((self.n, self.n))

        # This is a simplified version - in practice, this would use
        # the composite rigid body algorithm or articulated body algorithm

        # For each joint, calculate its contribution to the mass matrix
        for i in range(self.n):
            for j in range(self.n):
                H[i, j] = self.calculate_inertia_coupling(i, j, joint_angles)

        return H

    def coriolis_matrix(self, joint_angles, joint_velocities):
        """
        Calculate Coriolis matrix C(q, q_dot)
        """
        C = np.zeros((self.n, self.n))

        # Calculate Coriolis and centrifugal terms
        # This involves Christoffel symbols of the first kind
        H = self.mass_matrix(joint_angles)

        for i in range(self.n):
            for j in range(self.n):
                c_sum = 0
                for k in range(self.n):
                    # Christoffel symbol: Gamma^i_jk = 0.5 * (dH_ik/dq_j + dH_jk/dq_i - dH_ij/dq_k)
                    gamma_ijk = 0.5 * (
                        self.dH_dq(i, k, j, joint_angles) +
                        self.dH_dq(j, k, i, joint_angles) -
                        self.dH_dq(i, j, k, joint_angles)
                    )
                    c_sum += gamma_ijk * joint_velocities[k]
                C[i, j] = c_sum

        return C

    def gravity_vector(self, joint_angles):
        """
        Calculate gravity vector g(q)
        """
        g = np.zeros(self.n)

        # Calculate gravity torques for each joint
        # This depends on the robot's kinematic structure
        for i in range(self.n):
            g[i] = self.calculate_gravity_torque(i, joint_angles)

        return g

    def forward_dynamics(self, joint_angles, joint_velocities, torques):
        """
        Calculate joint accelerations from torques: H(q)q_ddot + C(q,q_dot)q_dot + g(q) = τ
        """
        H = self.mass_matrix(joint_angles)
        C = self.coriolis_matrix(joint_angles, joint_velocities)
        g = self.gravity_vector(joint_angles)

        # Solve: H * q_ddot = τ - C*q_dot - g
        q_ddot = np.linalg.solve(H, torques - C @ joint_velocities - g)
        return q_ddot

    def inverse_dynamics(self, joint_angles, joint_velocities, joint_accelerations):
        """
        Calculate required torques for desired motion: τ = H(q)q_ddot + C(q,q_dot)q_dot + g(q)
        """
        H = self.mass_matrix(joint_angles)
        C = self.coriolis_matrix(joint_angles, joint_velocities)
        g = self.gravity_vector(joint_angles)

        torques = H @ joint_accelerations + C @ joint_velocities + g
        return torques
```

### Newton-Euler Formulation

The Newton-Euler formulation is computationally more efficient for forward dynamics:

```python
def newton_euler_forward(robot_params, joint_angles, joint_velocities, joint_accelerations):
    """
    Forward dynamics using Newton-Euler formulation
    robot_params: List of link parameters [mass, com, inertia, joint_type, ...]
    """
    n = len(joint_angles)

    # Initialize link velocities and accelerations
    link_velocities = [np.zeros(6) for _ in range(n)]
    link_accelerations = [np.zeros(6) for _ in range(n)]
    link_forces = [np.zeros(6) for _ in range(n)]

    # Outward recursion: calculate velocities and accelerations
    for i in range(n):
        # Calculate link i velocity and acceleration
        if i == 0:
            # Base link (usually fixed or with known motion)
            link_velocities[i] = np.zeros(6)
            link_accelerations[i] = np.zeros(6)
        else:
            # Calculate from parent link
            parent_idx = robot_params[i]['parent']
            joint_type = robot_params[i]['joint_type']

            # Transform parent's motion to current link
            T = calculate_transform(robot_params, joint_angles[:i+1])
            link_velocities[i] = transform_twist(link_velocities[parent_idx], T)

            # Add joint motion contribution
            if joint_type == 'revolute':
                joint_axis = get_joint_axis(robot_params[i])
                joint_vel = joint_velocities[i] * joint_axis
                link_velocities[i][:3] += joint_vel  # Angular velocity
            else:  # prismatic
                joint_axis = get_joint_axis(robot_params[i])
                joint_vel = joint_velocities[i] * joint_axis
                link_velocities[i][3:] += joint_vel  # Linear velocity

    # Inward recursion: calculate forces and torques
    torques = np.zeros(n)
    for i in range(n-1, -1, -1):  # Start from end-effector
        # Calculate force and torque on link i
        link_mass = robot_params[i]['mass']
        link_inertia = robot_params[i]['inertia']

        # Apply Newton-Euler equations
        # F = m*a_com
        # T = I*alpha + omega x (I*omega)  (Euler's equation)

        # Extract linear and angular parts
        linear_acc = link_accelerations[i][3:]
        angular_acc = link_accelerations[i][:3]
        linear_vel = link_velocities[i][3:]
        angular_vel = link_velocities[i][:3]

        # Forces and torques
        force = link_mass * linear_acc
        torque = link_inertia @ angular_acc + np.cross(angular_vel, link_inertia @ angular_vel)

        # Calculate required joint torque/force
        joint_axis = get_joint_axis(robot_params[i])
        if robot_params[i]['joint_type'] == 'revolute':
            torques[i] = np.dot(torque, joint_axis)
        else:  # prismatic
            torques[i] = np.dot(force, joint_axis)

    return torques
```

## Humanoid-Specific Kinematics

### Humanoid Kinematic Structure

Humanoid robots have a complex kinematic structure that mimics human anatomy:

```python
class HumanoidRobot:
    def __init__(self):
        self.chains = {
            'right_arm': self.define_arm_chain('right'),
            'left_arm': self.define_arm_chain('left'),
            'right_leg': self.define_leg_chain('right'),
            'left_leg': self.define_leg_chain('left'),
            'torso': self.define_torso_chain(),
            'head': self.define_head_chain()
        }

        # Define kinematic constraints
        self.constraints = self.define_kinematic_constraints()

    def define_arm_chain(self, side):
        """Define kinematic chain for arm"""
        # Human-like arm with 7 DOF
        dh_params = [
            [0, np.pi/2, 0, 'revolute'],      # Shoulder Yaw (Tz)
            [0, -np.pi/2, 0, 'revolute'],     # Shoulder Pitch (Tx)
            [0, np.pi/2, 0, 'revolute'],      # Shoulder Roll (Ty)
            [0.2, 0, 0, 'revolute'],          # Elbow (Tx) - upper arm length
            [0, np.pi/2, 0, 'revolute'],      # Wrist Yaw (Tz)
            [0, -np.pi/2, 0, 'revolute'],     # Wrist Pitch (Tx)
            [0, 0, 0.1, 'revolute']           # Wrist Roll (Ty) - forearm length
        ]
        return ForwardKinematics(dh_params)

    def define_leg_chain(self, side):
        """Define kinematic chain for leg"""
        # Human-like leg with 6 DOF
        dh_params = [
            [0, np.pi/2, 0, 'revolute'],      # Hip Yaw (Tz)
            [0, -np.pi/2, 0, 'revolute'],     # Hip Roll (Tx) - can be zero for some robots
            [0, 0, 0, 'revolute'],            # Hip Pitch (Ty)
            [0, 0, -0.4, 'revolute'],         # Knee (Ty) - thigh length
            [0, 0, -0.4, 'revolute'],         # Ankle Pitch (Ty) - shank length
            [0, 0, 0, 'revolute']             # Ankle Roll (Tx)
        ]
        return ForwardKinematics(dh_params)

    def define_torso_chain(self):
        """Define torso kinematics (simplified as single joint)"""
        dh_params = [
            [0, 0, 0.2, 'revolute']  # Torso pitch (for upper body movement)
        ]
        return ForwardKinematics(dh_params)

    def define_head_chain(self):
        """Define head/neck kinematics"""
        dh_params = [
            [0, 0, 0, 'revolute'],    # Neck Yaw (Tz)
            [0, 0, 0, 'revolute'],    # Neck Pitch (Ty)
            [0, 0, 0, 'revolute']     # Neck Roll (Tx)
        ]
        return ForwardKinematics(dh_params)

    def calculate_full_body_pose(self, joint_angles_dict):
        """
        Calculate pose of all end-effectors given full body joint angles
        joint_angles_dict: Dictionary with chain names as keys and joint angles as values
        """
        poses = {}

        for chain_name, angles in joint_angles_dict.items():
            if chain_name in self.chains:
                fk = self.chains[chain_name]
                poses[chain_name] = fk.calculate_end_effector_pose(angles)

        return poses
```

### Whole-Body Kinematics

For humanoid robots, we often need to solve kinematics for multiple chains simultaneously:

```python
def whole_body_ik(
    humanoid_robot,
    task_descriptions,
    initial_configuration,
    weights=None
):
    """
    Solve whole-body inverse kinematics
    task_descriptions: List of tasks [(chain_name, target_pose, priority), ...]
    initial_configuration: Initial joint configuration
    weights: Weights for different task priorities
    """
    if weights is None:
        weights = [1.0] * len(task_descriptions)

    # This is a simplified implementation
    # In practice, whole-body IK often uses hierarchical or optimization-based approaches

    current_config = initial_configuration.copy()

    # Solve tasks in order of priority
    for i, (chain_name, target_pose, priority) in enumerate(task_descriptions):
        # For each task, solve IK while trying to maintain previous solutions
        chain = humanoid_robot.chains[chain_name]

        # Extract relevant joint angles for this chain
        chain_joints = get_chain_joints(chain_name)

        # Solve IK for this chain
        new_angles = inverse_kinematics_numerical(
            chain, target_pose, current_config[chain_joints]
        )

        # Update configuration
        current_config[chain_joints] = new_angles

    return current_config

def get_chain_joints(chain_name):
    """Get joint indices for a specific chain"""
    chain_joints_map = {
        'right_arm': slice(0, 7),      # Joints 0-6
        'left_arm': slice(7, 14),      # Joints 7-13
        'right_leg': slice(14, 20),    # Joints 14-19
        'left_leg': slice(20, 26),     # Joints 20-25
        'torso': slice(26, 27),        # Joint 26
        'head': slice(27, 30)          # Joints 27-29
    }
    return chain_joints_map[chain_name]
```

## Balance and Stability

### Center of Mass and Zero Moment Point

Balance is crucial for humanoid robots, especially during locomotion:

```python
def calculate_center_of_mass(robot, joint_angles, masses, com_positions):
    """
    Calculate center of mass of the robot
    robot: Robot model
    joint_angles: Current joint angles
    masses: Mass of each link
    com_positions: COM positions of each link in link frame
    """
    total_mass = sum(masses)

    # Calculate COM of each link in world frame
    link_poses = robot.calculate_all_link_poses(joint_angles)

    weighted_sum = np.zeros(3)
    for i, (mass, local_com, pose) in enumerate(zip(masses, com_positions, link_poses)):
        # Transform local COM to world frame
        world_com = pose[:3, :3] @ local_com + pose[:3, 3]
        weighted_sum += mass * world_com

    com = weighted_sum / total_mass
    return com

def calculate_zero_moment_point(robot, joint_angles, joint_velocities, joint_accelerations, masses, com_positions):
    """
    Calculate Zero Moment Point (ZMP) for balance assessment
    """
    # Calculate total wrench (force and moment) acting on the robot
    com = calculate_center_of_mass(robot, joint_angles, masses, com_positions)

    # Calculate acceleration of COM
    com_acc = calculate_com_acceleration(robot, joint_angles, joint_velocities, joint_accelerations)

    # Gravity vector
    g = np.array([0, 0, -9.81])

    # Net force on COM: F = m * (a_com - g)
    total_mass = sum(masses)
    net_force = total_mass * (com_acc - g)

    # ZMP calculation (assuming feet are on ground at z=0)
    if abs(net_force[2]) > 1e-6:  # Avoid division by zero
        zmp_x = com[0] - (com[2] * net_force[0]) / net_force[2]
        zmp_y = com[1] - (com[2] * net_force[1]) / net_force[2]
    else:
        zmp_x, zmp_y = com[0], com[1]

    return np.array([zmp_x, zmp_y, 0])

def calculate_com_acceleration(robot, joint_angles, joint_velocities, joint_accelerations):
    """
    Calculate acceleration of center of mass
    This is a simplified calculation
    """
    # In practice, this would require the derivative of the COM calculation
    # with respect to joint variables
    pass  # Implementation would depend on specific robot structure
```

### Balance Control Strategies

#### Inverted Pendulum Model

The linear inverted pendulum model (LIPM) is commonly used for humanoid balance:

```python
class BalanceController:
    def __init__(self, com_height, gravity=9.81):
        self.com_height = com_height
        self.omega = np.sqrt(gravity / com_height)  # Natural frequency
        self.gravity = gravity

    def calculate_capture_point(self, com_position, com_velocity):
        """
        Calculate capture point for balance control
        Capture point is where the robot should step to stop its motion
        """
        capture_point = com_position[:2] + com_velocity[:2] / self.omega
        return capture_point

    def generate_balance_pattern(self, start_pos, end_pos, step_time, dt):
        """
        Generate CoM trajectory for balance
        """
        t = np.arange(0, step_time, dt)

        # Use exponentially decaying trajectory
        x_traj = start_pos[0] + (end_pos[0] - start_pos[0]) * (1 - np.exp(-self.omega * t))
        y_traj = start_pos[1] + (end_pos[1] - start_pos[1]) * (1 - np.exp(-self.omega * t))

        # Calculate velocities
        vx_traj = (end_pos[0] - start_pos[0]) * self.omega * np.exp(-self.omega * t)
        vy_traj = (end_pos[1] - start_pos[1]) * self.omega * np.exp(-self.omega * t)

        return np.column_stack([x_traj, y_traj]), np.column_stack([vx_traj, vy_traj])

    def compute_balance_control(self, current_zmp, desired_zmp, current_com, current_com_vel):
        """
        Compute control for balance maintenance
        """
        # Simple PD control on ZMP error
        kp = 10.0  # Proportional gain
        kd = 2.0 * np.sqrt(kp)  # Derivative gain (critically damped)

        zmp_error = desired_zmp - current_zmp
        com_control = kp * zmp_error + kd * (-current_com_vel[:2])  # Negative velocity for damping

        return com_control
```

### Walking Pattern Generation

```python
class WalkingPatternGenerator:
    def __init__(self, step_length=0.3, step_width=0.2, step_height=0.05, step_time=1.0):
        self.step_length = step_length
        self.step_width = step_width
        self.step_height = step_height
        self.step_time = step_time

    def generate_foot_trajectory(self, start_pos, end_pos, support_foot='left', dt=0.01):
        """
        Generate foot trajectory for stepping
        """
        t = np.arange(0, self.step_time, dt)
        n_steps = len(t)

        # Generate trajectory using 3rd order polynomial for smooth motion
        # x, y: linear interpolation
        x_traj = np.linspace(start_pos[0], end_pos[0], n_steps)
        y_traj = np.linspace(start_pos[1], end_pos[1], n_steps)

        # z: 3rd order polynomial for smooth lift and place
        z_lift = np.zeros(n_steps)
        for i, ti in enumerate(t):
            if ti < self.step_time / 3:  # Lift phase
                ratio = ti / (self.step_time / 3)
                z_lift[i] = self.step_height * (3*ratio**2 - 2*ratio**3)
            elif ti > 2 * self.step_time / 3:  # Place phase
                ratio = (ti - 2*self.step_time/3) / (self.step_time/3)
                z_lift[i] = self.step_height * (1 - (3*(1-ratio)**2 - 2*(1-ratio)**3))
            else:  # Constant height phase
                z_lift[i] = self.step_height

        return np.column_stack([x_traj, y_traj, z_lift])

    def generate_walking_pattern(self, steps, start_pos=np.array([0, 0]), start_support='left'):
        """
        Generate complete walking pattern for multiple steps
        """
        pattern = []
        current_pos = start_pos.copy()
        support_foot = start_support

        for i, step in enumerate(steps):
            # Determine next foot position
            if support_foot == 'left':
                next_pos = current_pos + np.array([self.step_length, -self.step_width/2])
                swing_foot = 'right'
            else:
                next_pos = current_pos + np.array([self.step_length, self.step_width/2])
                swing_foot = 'left'

            # Generate foot trajectory
            foot_traj = self.generate_foot_trajectory(
                current_pos, next_pos, support_foot
            )

            pattern.append({
                'step': i,
                'swing_foot': swing_foot,
                'support_foot': support_foot,
                'trajectory': foot_traj,
                'start_pos': current_pos.copy(),
                'end_pos': next_pos.copy()
            })

            # Update for next step
            current_pos = next_pos
            support_foot = swing_foot

        return pattern
```

## Implementation Considerations

### Numerical Stability

Kinematics and dynamics calculations require careful attention to numerical stability:

```python
def robust_rotation_matrix_to_euler(R, convention='xyz'):
    """
    Robust conversion from rotation matrix to Euler angles
    Handles singularities and numerical errors
    """
    if convention == 'xyz':
        # Handle gimbal lock near pitch = ±π/2
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)

        singular = sy < 1e-6  # If R[2,0] is close to ±1

        if not singular:
            x = np.arctan2(R[2,1], R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0

        return np.array([x, y, z])
    else:
        # Other conventions can be added as needed
        raise NotImplementedError(f"Convention {convention} not implemented")

def robust_quaternion_to_rotation_matrix(q):
    """
    Robust conversion from quaternion to rotation matrix
    Normalizes quaternion to handle numerical errors
    """
    q = q / np.linalg.norm(q)  # Normalize to handle numerical errors

    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z

    if Nq < 1e-8:
        return np.eye(3)

    s = 2.0 / Nq
    X = x*s; Y = y*s; Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z

    R = np.array([
        [1.0-(yY+zZ), xY-wZ, xZ+wY],
        [xY+wZ, 1.0-(xX+zZ), yZ-wX],
        [xZ-wY, yZ+wX, 1.0-(xX+yY)]
    ])
    return R

def check_jacobian_condition(J, threshold=1e-6):
    """
    Check condition number of Jacobian to detect singularities
    """
    U, s, Vt = np.linalg.svd(J)
    condition_number = s[0] / (s[-1] + 1e-12)  # Add small value to avoid division by zero

    is_singular = s[-1] < threshold

    return condition_number, is_singular
```

### Performance Optimization

For real-time humanoid control, performance optimization is crucial:

```python
import numba
from numba import jit

@jit(nopython=True)
def fast_dh_transform(a, alpha, d, theta):
    """
    Fast DH transformation using Numba JIT compilation
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    T = np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    return T

@jit(nopython=True)
def fast_forward_kinematics(dh_params, joint_angles):
    """
    Fast forward kinematics using JIT compilation
    """
    T_total = np.eye(4, dtype=np.float64)

    for i in range(len(joint_angles)):
        a, alpha, d = dh_params[i, 0], dh_params[i, 1], dh_params[i, 2]
        theta = joint_angles[i] if dh_params[i, 3] == 1 else 0  # 1 for revolute, 0 for prismatic

        T_link = fast_dh_transform(a, alpha, d, theta)
        T_total = T_total @ T_link

    return T_total

class OptimizedKinematics:
    def __init__(self, dh_params):
        self.dh_params = np.array(dh_params, dtype=np.float64)
        self.n = len(dh_params)

    def calculate_pose(self, joint_angles):
        """Calculate end-effector pose with optimized computation"""
        joint_array = np.array(joint_angles, dtype=np.float64)
        return fast_forward_kinematics(self.dh_params, joint_array)
```

### Safety and Validation

Humanoid robots require careful validation of kinematic and dynamic calculations:

```python
class KinematicValidator:
    def __init__(self, robot_model):
        self.model = robot_model
        self.joint_limits = robot_model.joint_limits

    def validate_joint_configuration(self, joint_angles):
        """Validate that joint angles are within limits"""
        joint_angles = np.array(joint_angles)

        if len(joint_angles) != len(self.joint_limits):
            return False, "Wrong number of joint angles"

        for i, (angle, (min_limit, max_limit)) in enumerate(zip(joint_angles, self.joint_limits)):
            if angle < min_limit or angle > max_limit:
                return False, f"Joint {i} out of limits: {angle} not in [{min_limit}, {max_limit}]"

        return True, "Valid configuration"

    def validate_reachability(self, target_pose, base_pose=np.eye(4)):
        """Validate that target pose is reachable"""
        # Calculate maximum reach (simplified)
        max_reach = self.calculate_max_reach()

        # Calculate distance from base to target
        target_pos = target_pose[:3, 3]
        base_pos = base_pose[:3, 3]
        distance = np.linalg.norm(target_pos - base_pos)

        if distance > max_reach:
            return False, f"Target out of reach: {distance:.3f} > {max_reach:.3f}"

        return True, "Reachable target"

    def calculate_max_reach(self):
        """Calculate maximum reach of the robot"""
        # Sum of all link lengths (simplified)
        total_length = 0
        for dh_params in self.model.dh_params:
            a, alpha, d, joint_type = dh_params
            total_length += abs(a) + abs(d)

        return total_length

    def validate_kinematic_solution(self, joint_angles, target_pose, tolerance=1e-3):
        """Validate that a kinematic solution achieves the target"""
        current_pose = self.model.calculate_pose(joint_angles)

        # Check position error
        pos_error = np.linalg.norm(target_pose[:3, 3] - current_pose[:3, 3])

        # Check orientation error
        R_error = target_pose[:3, :3] @ current_pose[:3, :3].T
        trace = np.trace(R_error)
        angle_error = np.arccos(np.clip((trace - 1) / 2, -1, 1))

        if pos_error > tolerance or angle_error > tolerance:
            return False, f"Solution error too large: pos={pos_error:.6f}, rot={angle_error:.6f}"

        return True, "Valid solution"
```

## Exercises

1. Implement forward kinematics for a 6-DOF robotic arm and verify your implementation by plotting the workspace.

2. Solve the inverse kinematics for a 3-DOF planar manipulator using both analytical and numerical methods. Compare the results.

3. Implement the Jacobian calculation for a simple 2-DOF arm and use it to control the end-effector velocity.

4. Derive the dynamic equations for a simple pendulum and implement a simulation to verify your results.

5. Calculate the center of mass for a simplified humanoid model and implement a balance controller.

6. Implement a walking pattern generator for a bipedal robot and simulate the resulting motion.

7. Design a whole-body IK solver for a humanoid robot that can simultaneously control both hands and maintain balance.

8. Implement a capture point-based balance controller and test it with perturbation experiments.

9. Optimize your kinematic calculations using JIT compilation or other performance techniques.

10. Validate your kinematic and dynamic implementations using a physics simulator.

## Next Steps

After completing this chapter, you should have a solid understanding of kinematics and dynamics for humanoid robots. The next chapter will explore bipedal locomotion in detail, building upon the kinematic and dynamic foundations established here.