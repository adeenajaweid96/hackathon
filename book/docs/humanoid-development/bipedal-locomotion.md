# Bipedal Locomotion

## Overview

Bipedal locomotion is one of the most challenging aspects of humanoid robotics, requiring sophisticated control strategies to achieve stable, efficient, and human-like walking. Unlike wheeled or tracked robots, bipedal robots must continuously balance themselves while moving, making locomotion a complex interplay of kinematics, dynamics, balance control, and environmental adaptation.

This chapter explores the principles, techniques, and implementation strategies for achieving stable bipedal locomotion. We'll examine various walking pattern generation methods, balance control strategies, and the integration of perception for adaptive locomotion in complex environments.

## Learning Objectives

By the end of this chapter, you should be able to:

- Understand the biomechanics and physics of human walking
- Implement various bipedal walking pattern generation techniques
- Design and implement balance control systems for walking
- Apply Zero Moment Point (ZMP) theory for stable walking
- Generate walking patterns using preview control
- Implement adaptive locomotion for different terrains
- Understand the relationship between walking speed, stability, and energy efficiency
- Evaluate walking performance and stability metrics

## Table of Contents

1. [Introduction to Bipedal Locomotion](#introduction-to-bipedal-locomotion)
2. [Human Walking Biomechanics](#human-walking-biomechanics)
3. [Walking Pattern Generation](#walking-pattern-generation)
4. [Balance Control for Walking](#balance-control-for-walking)
5. [ZMP-Based Walking Control](#zmp-based-walking-control)
6. [Preview Control for Walking](#preview-control-for-walking)
7. [Adaptive and Robust Walking](#adaptive-and-robust-walking)
8. [Implementation Considerations](#implementation-considerations)
9. [Exercises](#exercises)

## Introduction to Bipedal Locomotion

### The Challenge of Bipedal Walking

Bipedal walking presents unique challenges compared to other forms of locomotion:

- **Dynamic Balance**: Unlike static structures, walking requires continuous balance adjustment
- **Underactuation**: The robot is essentially an inverted pendulum that needs active control
- **Multi-contact dynamics**: The transition between single and double support phases
- **Energy efficiency**: Achieving human-like efficiency while maintaining stability
- **Terrain adaptation**: Handling various surfaces and obstacles

### Walking Phases and Gait Cycle

Human walking consists of two main phases:

#### Stance Phase (60% of gait cycle)
- **Double Support**: Both feet on ground (about 20% of cycle)
- **Single Support**: Weight on one foot (about 40% of cycle)

#### Swing Phase (40% of gait cycle)
- **Pre-swing**: Push-off from stance leg
- **Initial swing**: Foot clearance
- **Mid-swing**: Limb advancement
- **Terminal swing**: Foot placement preparation

### Key Walking Parameters

#### Temporal Parameters
- **Step time**: Time for one complete step cycle
- **Stride time**: Time for two consecutive steps
- **Cadence**: Steps per minute

#### Spatial Parameters
- **Step length**: Distance between consecutive foot placements
- **Stride length**: Distance between two consecutive placements of the same foot
- **Step width**: Lateral distance between feet

#### Dynamic Parameters
- **Walking speed**: Forward velocity
- **Ground reaction forces**: Forces exerted by the ground
- **Joint torques**: Required actuator forces

## Human Walking Biomechanics

### Center of Mass Motion

Human walking is characterized by a specific center of mass (CoM) motion pattern:

```python
import numpy as np
import matplotlib.pyplot as plt

def human_com_pattern(gait_cycle, step_length, com_height):
    """
    Generate human-like CoM motion pattern
    gait_cycle: Array of gait cycle percentages (0 to 1)
    step_length: Length of one step
    com_height: Height of center of mass
    """
    # Lateral CoM motion (side-to-side sway)
    lateral_sway = 0.02 * np.sin(2 * np.pi * gait_cycle)  # 2cm sway

    # Forward CoM motion (speed variations)
    forward_motion = step_length * gait_cycle + 0.01 * np.sin(4 * np.pi * gait_cycle)  # Forward with slight oscillation

    # Vertical CoM motion (up-down movement)
    vertical_motion = com_height + 0.015 * np.cos(2 * np.pi * gait_cycle)  # 1.5cm up-down

    return np.column_stack([forward_motion, lateral_sway, vertical_motion])

# Example usage
gait_cycle = np.linspace(0, 1, 100)
com_trajectory = human_com_pattern(gait_cycle, 0.7, 0.85)  # 70cm step, 85cm CoM height
```

### Joint Coordination Patterns

Human walking exhibits specific coordination patterns between joints:

```python
def generate_human_joint_angles(gait_cycle, walking_speed=1.0):
    """
    Generate human-like joint angle patterns during walking
    gait_cycle: Array of gait cycle percentages (0 to 1)
    walking_speed: Walking speed in m/s
    """
    # Normalize gait cycle to 0-2π for trigonometric functions
    phase = 2 * np.pi * gait_cycle

    # Hip angles (sagittal plane)
    stance_hip = -0.1 * np.cos(phase)  # Hip extension during stance
    swing_hip = 0.3 * np.sin(phase) + 0.2  # Hip flexion during swing

    # Knee angles (sagittal plane)
    stance_knee = 0.05 * np.cos(phase)  # Slight knee flexion during stance
    swing_knee = -0.6 * np.sin(phase) + 0.3  # Knee flexion during swing

    # Ankle angles (sagittal plane)
    stance_ankle = 0.1 * np.sin(phase)  # Ankle rocker during stance
    swing_ankle = 0.1  # Neutral position during swing

    # Combine based on gait phase
    hip_angle = np.where(gait_cycle < 0.6, stance_hip, swing_hip)
    knee_angle = np.where(gait_cycle < 0.6, stance_knee, swing_knee)
    ankle_angle = np.where(gait_cycle < 0.6, stance_ankle, swing_ankle)

    return hip_angle, knee_angle, ankle_angle
```

### Ground Reaction Forces

Ground reaction forces during walking follow characteristic patterns:

```python
def ground_reaction_forces(gait_cycle):
    """
    Generate typical ground reaction force patterns during walking
    """
    # Vertical GRF (double peak pattern)
    first_peak = 1.15 * np.exp(-((gait_cycle - 0.15) / 0.1)**2)  # Heel strike
    second_peak = 1.25 * np.exp(-((gait_cycle - 0.45) / 0.1)**2)  # Push-off
    vertical_force = 0.9 + first_peak + second_peak  # Body weight + peaks

    # Anterior-posterior GRF (braking and propulsive forces)
    braking_force = -0.25 * np.exp(-((gait_cycle - 0.15) / 0.1)**2)  # Braking in early stance
    propulsive_force = 0.2 * np.exp(-((gait_cycle - 0.45) / 0.1)**2)  # Propulsion in late stance
    horizontal_force = braking_force + propulsive_force

    # Medial-lateral GRF (minimal in normal walking)
    lateral_force = 0.05 * np.sin(2 * np.pi * gait_cycle)

    return np.column_stack([horizontal_force, lateral_force, vertical_force])
```

## Walking Pattern Generation

### Inverted Pendulum Model

The inverted pendulum model is fundamental to understanding bipedal walking:

```python
class InvertedPendulumWalking:
    def __init__(self, com_height, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / com_height)

    def calculate_zmp_trajectory(self, com_trajectory, com_velocity, com_acceleration):
        """
        Calculate ZMP trajectory from CoM motion
        """
        zmp_x = com_trajectory[:, 0] - (com_acceleration[:, 0] - self.gravity) / (self.gravity / self.com_height)
        zmp_y = com_trajectory[:, 1] - (com_acceleration[:, 1] - self.gravity) / (self.gravity / self.com_height)

        return np.column_stack([zmp_x, zmp_y, np.zeros_like(zmp_x)])

    def generate_com_trajectory(self, start_pos, end_pos, step_time, dt):
        """
        Generate CoM trajectory using 3rd order polynomial
        """
        t = np.arange(0, step_time, dt)
        n_points = len(t)

        # 3rd order polynomial: x(t) = a0 + a1*t + a2*t^2 + a3*t^3
        # Boundary conditions: x(0)=start, x(T)=end, dx(0)=0, dx(T)=0

        T = step_time
        a0 = start_pos
        a1 = 0  # Initial velocity = 0
        a2 = 3 * (end_pos - start_pos) / T**2
        a3 = -2 * (end_pos - start_pos) / T**3

        x_traj = a0 + a1*t + a2*t**2 + a3*t**3
        vx_traj = a1 + 2*a2*t + 3*a3*t**2
        ax_traj = 2*a2 + 6*a3*t

        return x_traj, vx_traj, ax_traj
```

### Linear Inverted Pendulum Model (LIPM)

The LIPM simplifies walking by keeping CoM height constant:

```python
class LinearInvertedPendulumModel:
    def __init__(self, com_height, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / com_height)

    def calculate_foot_placement(self, com_pos, com_vel):
        """
        Calculate foot placement to stop the robot's motion (capture point)
        """
        capture_point_x = com_pos[0] + com_vel[0] / self.omega
        capture_point_y = com_pos[1] + com_vel[1] / self.omega

        return np.array([capture_point_x, capture_point_y])

    def generate_swing_foot_trajectory(self, start_pos, end_pos, step_height=0.05, dt=0.01):
        """
        Generate trajectory for swing foot
        """
        step_time = 0.8  # Typical step time
        t = np.arange(0, step_time, dt)
        n_points = len(t)

        # X and Y trajectories (simple interpolation)
        x_traj = np.linspace(start_pos[0], end_pos[0], n_points)
        y_traj = np.linspace(start_pos[1], end_pos[1], n_points)

        # Z trajectory (parabolic lift and place)
        z_traj = np.zeros(n_points)
        for i, ti in enumerate(t):
            if ti < step_time / 3:  # Lift phase
                ratio = ti / (step_time / 3)
                z_traj[i] = step_height * (3*ratio**2 - 2*ratio**3)
            elif ti > 2 * step_time / 3:  # Place phase
                ratio = (ti - 2*step_time/3) / (step_time/3)
                z_traj[i] = step_height * (1 - (3*(1-ratio)**2 - 2*(1-ratio)**3))
            else:  # Constant height phase
                z_traj[i] = step_height

        return np.column_stack([x_traj, y_traj, z_traj])

    def generate_com_trajectory(self, zmp_trajectory, dt):
        """
        Generate CoM trajectory from ZMP reference using LIPM
        """
        n_points = len(zmp_trajectory)
        com_trajectory = np.zeros((n_points, 3))
        com_velocity = np.zeros((n_points, 3))
        com_acceleration = np.zeros((n_points, 3))

        # Initial conditions
        com_trajectory[0, :2] = zmp_trajectory[0, :2]  # Start at ZMP
        com_trajectory[0, 2] = self.com_height  # Constant height

        # Integrate LIPM equations: COM_ddot = omega^2 * (COM - ZMP)
        for i in range(1, n_points):
            # Acceleration based on ZMP error
            zmp_error = com_trajectory[i-1, :2] - zmp_trajectory[i-1, :2]
            com_acceleration[i, :2] = self.omega**2 * zmp_error

            # Update velocity and position
            com_velocity[i, :2] = com_velocity[i-1, :2] + com_acceleration[i, :2] * dt
            com_trajectory[i, :2] = com_trajectory[i-1, :2] + com_velocity[i, :2] * dt

            # Keep height constant
            com_trajectory[i, 2] = self.com_height

        return com_trajectory, com_velocity, com_acceleration
```

### Walking Pattern Generation Algorithms

#### Footstep Planning

```python
class FootstepPlanner:
    def __init__(self, step_length=0.3, step_width=0.2, max_turn=0.2):
        self.step_length = step_length
        self.step_width = step_width
        self.max_turn = max_turn

    def plan_footsteps(self, path, start_pose, support_foot='left'):
        """
        Plan footstep locations along a path
        path: List of (x, y, theta) waypoints
        start_pose: Initial robot pose
        support_foot: Initial support foot ('left' or 'right')
        """
        footsteps = []
        current_pose = start_pose.copy()
        support_foot_type = support_foot

        for i, target_pose in enumerate(path):
            # Calculate required step
            dx = target_pose[0] - current_pose[0]
            dy = target_pose[1] - current_pose[1]
            dtheta = target_pose[2] - current_pose[2]

            # Limit step size and turn
            step_length = min(np.sqrt(dx**2 + dy**2), self.step_length)
            step_angle = min(abs(dtheta), self.max_turn)

            # Calculate step direction
            step_direction = np.arctan2(dy, dx)
            step_x = current_pose[0] + step_length * np.cos(step_direction + current_pose[2])
            step_y = current_pose[1] + step_length * np.sin(step_direction + current_pose[2])
            step_theta = current_pose[2] + step_angle * np.sign(dtheta)

            # Determine swing foot position
            if support_foot_type == 'left':
                # Right foot steps forward
                swing_x = step_x
                swing_y = step_y - self.step_width/2
                next_support = 'right'
            else:
                # Left foot steps forward
                swing_x = step_x
                swing_y = step_y + self.step_width/2
                next_support = 'left'

            footsteps.append({
                'step_num': i,
                'swing_foot': 'right' if support_foot_type == 'left' else 'left',
                'position': (swing_x, swing_y, step_theta),
                'support_foot': support_foot_type
            })

            # Update for next step
            current_pose = np.array([step_x, step_y, step_theta])
            support_foot_type = next_support

        return footsteps
```

#### Walking Pattern Generator

```python
class WalkingPatternGenerator:
    def __init__(self, com_height=0.8, step_length=0.3, step_width=0.2, step_time=0.8):
        self.com_height = com_height
        self.step_length = step_length
        self.step_width = step_width
        self.step_time = step_time
        self.lipm = LinearInvertedPendulumModel(com_height)

    def generate_walk_pattern(self, footsteps, dt=0.01):
        """
        Generate complete walking pattern from footsteps
        """
        n_steps = len(footsteps)
        total_time = n_steps * self.step_time
        n_points = int(total_time / dt)

        # Initialize trajectories
        com_trajectory = np.zeros((n_points, 3))
        zmp_trajectory = np.zeros((n_points, 3))
        foot_trajectory = np.zeros((n_points, 3, 2))  # [time, xyz, left/right]

        current_time = 0
        current_com = np.array([0, 0, self.com_height])
        current_zmp = np.array([0, 0, 0])

        for i, step in enumerate(footsteps):
            step_start_idx = int(current_time / dt)
            step_end_idx = int((current_time + self.step_time) / dt)

            # Calculate step parameters
            swing_foot_pos = step['position']
            support_foot = step['support_foot']

            # Generate ZMP trajectory for this step
            zmp_ref = self.generate_step_zmp_trajectory(
                current_com[:2], swing_foot_pos[:2], dt
            )

            # Generate CoM trajectory following ZMP reference
            com_step, _, _ = self.lipm.generate_com_trajectory(
                zmp_ref, dt
            )

            # Generate swing foot trajectory
            if support_foot == 'left':
                # Right foot is swing foot
                support_foot_pos = np.array([current_com[0], -self.step_width/2, 0])
                swing_trajectory = self.lipm.generate_swing_foot_trajectory(
                    support_foot_pos, swing_foot_pos[:3], dt=dt
                )
                foot_trajectory[step_start_idx:step_end_idx, :, 0] = support_foot_pos  # Left foot
                foot_trajectory[step_start_idx:step_end_idx, :, 1] = swing_trajectory  # Right foot
            else:
                # Left foot is swing foot
                support_foot_pos = np.array([current_com[0], self.step_width/2, 0])
                swing_trajectory = self.lipm.generate_swing_foot_trajectory(
                    support_foot_pos, swing_foot_pos[:3], dt=dt
                )
                foot_trajectory[step_start_idx:step_end_idx, :, 0] = swing_trajectory  # Left foot
                foot_trajectory[step_start_idx:step_end_idx, :, 1] = support_foot_pos  # Right foot

            # Update trajectories
            com_trajectory[step_start_idx:step_end_idx, :] = com_step
            zmp_trajectory[step_start_idx:step_end_idx, :2] = zmp_ref[:, :2]

            # Update for next step
            current_com[:2] = swing_foot_pos[:2]
            current_time += self.step_time

        return {
            'com_trajectory': com_trajectory,
            'zmp_trajectory': zmp_trajectory,
            'foot_trajectories': foot_trajectory,
            'timestamps': np.arange(0, total_time, dt)
        }

    def generate_step_zmp_trajectory(self, start_com, target_foot, dt):
        """
        Generate ZMP trajectory for a single step
        """
        step_duration = self.step_time
        t = np.arange(0, step_duration, dt)
        n_points = len(t)

        # Simple ZMP pattern: start at CoM, end at foot position
        zmp_x = np.linspace(start_com[0], target_foot[0], n_points)
        zmp_y = np.linspace(start_com[1], target_foot[1], n_points)

        # Add small adjustments for natural walking pattern
        # Apply smoothing to make ZMP transition gradual
        zmp_x_smooth = self.smooth_zmp_transition(zmp_x, start_com[0], target_foot[0])
        zmp_y_smooth = self.smooth_zmp_transition(zmp_y, start_com[1], target_foot[1])

        return np.column_stack([zmp_x_smooth, zmp_y_smooth, np.zeros(n_points)])

    def smooth_zmp_transition(self, zmp_values, start_val, end_val):
        """
        Smooth ZMP transition with appropriate profile
        """
        n = len(zmp_values)
        # Apply 5th order polynomial smoothing
        t = np.linspace(0, 1, n)
        smooth_curve = (10*t**3 - 15*t**4 + 6*t**5)  # Smooth S-curve
        return start_val + (end_val - start_val) * smooth_curve
```

## Balance Control for Walking

### Feedback Control Strategies

#### PD Control for Balance

```python
class BalanceController:
    def __init__(self, com_height, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / com_height)

        # PD gains for balance control
        self.kp_com = 10.0  # Proportional gain for CoM error
        self.kd_com = 2.0 * np.sqrt(self.kp_com)  # Derivative gain (critically damped)
        self.kp_zmp = 5.0   # Proportional gain for ZMP error
        self.kd_zmp = 2.0 * np.sqrt(self.kp_zmp)  # Derivative gain

    def compute_balance_control(self, current_com, desired_com, current_com_vel, desired_com_vel):
        """
        Compute balance control corrections
        """
        # CoM position error
        com_error = desired_com - current_com
        com_vel_error = desired_com_vel - current_com_vel

        # Balance control (move ZMP to correct CoM error)
        zmp_correction = self.kp_com * com_error[:2] + self.kd_com * com_vel_error[:2]

        return zmp_correction

    def compute_foot_placement_correction(self, zmp_error, com_pos, com_vel):
        """
        Compute foot placement correction based on ZMP error
        """
        # Use capture point concept to determine where to step
        capture_point = self.calculate_capture_point(com_pos, com_vel)

        # If ZMP error is large, adjust foot placement toward capture point
        if np.linalg.norm(zmp_error) > 0.05:  # 5cm threshold
            # Move foot placement toward capture point
            correction = 0.3 * (capture_point - com_pos[:2])  # 30% toward capture point
            return correction
        else:
            return np.zeros(2)

    def calculate_capture_point(self, com_pos, com_vel):
        """
        Calculate capture point where robot should step to stop
        """
        capture_point_x = com_pos[0] + com_vel[0] / self.omega
        capture_point_y = com_pos[1] + com_vel[1] / self.omega

        return np.array([capture_point_x, capture_point_y])
```

### Model Predictive Control (MPC) for Walking

```python
import cvxpy as cp

class WalkingMPCController:
    def __init__(self, com_height, prediction_horizon=20, dt=0.01):
        self.com_height = com_height
        self.gravity = 9.81
        self.omega = np.sqrt(self.gravity / com_height)
        self.N = prediction_horizon
        self.dt = dt

    def setup_optimization_problem(self, Q, R, P):
        """
        Set up MPC optimization problem
        Q: State cost matrix
        R: Control cost matrix
        P: Terminal cost matrix
        """
        # State: [x, y, x_dot, y_dot] (CoM position and velocity)
        # Control: [zmp_x, zmp_y] (desired ZMP)
        n_states = 4  # x, y, x_dot, y_dot
        n_controls = 2  # zmp_x, zmp_y

        # Define variables
        X = cp.Variable((n_states, self.N + 1))  # State trajectory
        U = cp.Variable((n_controls, self.N))    # Control trajectory

        # System dynamics: x[k+1] = A*x[k] + B*u[k] + c
        A = self.get_system_matrix()
        B = self.get_input_matrix()
        c = np.zeros(n_states)  # Simplified model

        # Cost function
        cost = 0
        for k in range(self.N):
            cost += cp.quad_form(X[:, k], Q) + cp.quad_form(U[:, k], R)
        cost += cp.quad_form(X[:, self.N], P)  # Terminal cost

        # Constraints
        constraints = []
        for k in range(self.N):
            constraints.append(X[:, k+1] == A @ X[:, k] + B @ U[:, k] + c)

        # ZMP limits
        zmp_limits = 0.1  # 10cm limits
        for k in range(self.N):
            constraints += [cp.abs(U[0, k]) <= zmp_limits, cp.abs(U[1, k]) <= zmp_limits]

        # Initial state constraint
        x_init = cp.Parameter(n_states)
        constraints.append(X[:, 0] == x_init)

        # Create problem
        problem = cp.Problem(cp.Minimize(cost), constraints)

        return problem, X, U, x_init

    def get_system_matrix(self):
        """
        Get discrete-time system matrix for LIPM
        """
        A = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [self.omega**2 * self.dt, 0, 1, 0],
            [0, self.omega**2 * self.dt, 0, 1]
        ])
        return A

    def get_input_matrix(self):
        """
        Get discrete-time input matrix for LIPM
        """
        B = np.array([
            [0, 0],
            [0, 0],
            [-self.omega**2 * self.dt, 0],
            [0, -self.omega**2 * self.dt]
        ])
        return B

    def compute_control(self, current_state, reference_trajectory):
        """
        Compute optimal control using MPC
        """
        # Set up problem
        Q = np.eye(4) * 1.0  # State tracking cost
        R = np.eye(2) * 0.1  # Control effort cost
        P = np.eye(4) * 5.0  # Terminal cost

        problem, X, U, x_init = self.setup_optimization_problem(Q, R, P)

        # Set initial state
        x_init.value = current_state

        # Solve problem
        problem.solve(solver=cp.ECOS)

        if problem.status == cp.OPTIMAL:
            # Return first control input
            return U[:, 0].value
        else:
            # Fallback: return zero control
            return np.zeros(2)
```

## ZMP-Based Walking Control

### Zero Moment Point Theory

The Zero Moment Point (ZMP) is a critical concept in bipedal walking control:

```python
class ZMPController:
    def __init__(self, com_height, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / com_height)

    def calculate_zmp(self, com_pos, com_acc):
        """
        Calculate ZMP from CoM position and acceleration
        ZMP_x = CoM_x - (CoM_height * CoM_acc_x) / gravity
        ZMP_y = CoM_y - (CoM_height * CoM_acc_y) / gravity
        """
        zmp_x = com_pos[0] - (self.com_height * com_acc[0]) / self.gravity
        zmp_y = com_pos[1] - (self.com_height * com_acc[1]) / self.gravity

        return np.array([zmp_x, zmp_y, 0.0])

    def is_stable(self, zmp_pos, support_polygon):
        """
        Check if ZMP is within support polygon (stability test)
        support_polygon: List of (x, y) vertices of support polygon
        """
        # For a rectangular support polygon (simple case)
        if len(support_polygon) == 4:
            # Calculate bounds
            x_coords = [p[0] for p in support_polygon]
            y_coords = [p[1] for p in support_polygon]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Check if ZMP is within bounds
            is_inside = (x_min <= zmp_pos[0] <= x_max and
                        y_min <= zmp_pos[1] <= y_max)
            return is_inside

        # For general polygon, use point-in-polygon test
        return self.point_in_polygon(zmp_pos[:2], support_polygon)

    def point_in_polygon(self, point, polygon):
        """
        Ray casting algorithm to check if point is in polygon
        """
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def calculate_stability_margin(self, zmp_pos, support_polygon):
        """
        Calculate stability margin (distance from ZMP to polygon boundary)
        """
        # Calculate minimum distance from ZMP to polygon edges
        min_distance = float('inf')

        for i in range(len(support_polygon)):
            p1 = support_polygon[i]
            p2 = support_polygon[(i + 1) % len(support_polygon)]

            # Calculate distance from point to line segment
            distance = self.point_to_line_distance(zmp_pos[:2], p1, p2)
            min_distance = min(min_distance, distance)

        return min_distance

    def point_to_line_distance(self, point, line_start, line_end):
        """
        Calculate distance from point to line segment
        """
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end

        # Vector from line_start to point
        A = x - x1
        B = y - y1

        # Vector from line_start to line_end
        C = x2 - x1
        D = y2 - y1

        # Dot product
        dot = A * C + B * D
        len_sq = C * C + D * D

        if len_sq == 0:
            # Line segment is actually a point
            return np.sqrt(A * A + B * B)

        # Parameter of closest point on line
        param = dot / len_sq

        if param < 0:
            # Closest point is line_start
            xx = x1
            yy = y1
        elif param > 1:
            # Closest point is line_end
            xx = x2
            yy = y2
        else:
            # Closest point is on the segment
            xx = x1 + param * C
            yy = y1 + param * D

        dx = x - xx
        dy = y - yy

        return np.sqrt(dx * dx + dy * dy)
```

### ZMP Trajectory Planning

```python
class ZMPTrajectoryPlanner:
    def __init__(self, com_height, step_length=0.3, step_width=0.2, step_time=0.8):
        self.com_height = com_height
        self.step_length = step_length
        self.step_width = step_width
        self.step_time = step_time
        self.gravity = 9.81
        self.omega = np.sqrt(self.gravity / com_height)

    def plan_zmp_trajectory(self, footsteps, dt=0.01):
        """
        Plan ZMP trajectory following footsteps
        """
        total_time = len(footsteps) * self.step_time
        t = np.arange(0, total_time, dt)
        n_points = len(t)

        zmp_trajectory = np.zeros((n_points, 3))

        for i, step in enumerate(footsteps):
            step_start_time = i * self.step_time
            step_end_time = (i + 1) * self.step_time

            # Find indices for this step
            step_indices = np.where((t >= step_start_time) & (t < step_end_time))[0]

            if len(step_indices) == 0:
                continue

            # Get support foot for this step
            support_foot = step['support_foot']
            swing_foot_pos = step['position']

            # Calculate support polygon
            if support_foot == 'left':
                support_pos = np.array([step_start_time * self.step_length, self.step_width/2, 0])
            else:
                support_pos = np.array([step_start_time * self.step_length, -self.step_width/2, 0])

            # Plan ZMP trajectory for this step
            # During single support: ZMP moves toward next foot position
            # During double support: ZMP transitions between feet
            for j, idx in enumerate(step_indices):
                current_time_in_step = t[idx] - step_start_time
                phase_ratio = current_time_in_step / self.step_time

                if phase_ratio < 0.2:  # Early stance (ZMP near current foot)
                    zmp_trajectory[idx, :2] = support_pos[:2]
                elif phase_ratio > 0.8:  # Late stance (ZMP moving to next foot)
                    zmp_trajectory[idx, :2] = (
                        support_pos[:2] * (0.8 - phase_ratio) / 0.2 +
                        swing_foot_pos[:2] * (phase_ratio - 0.8) / 0.2
                    )
                else:  # Mid-stance (ZMP follows smooth transition)
                    zmp_trajectory[idx, :2] = support_pos[:2]

        return zmp_trajectory

    def generate_com_from_zmp(self, zmp_trajectory, dt):
        """
        Generate CoM trajectory from ZMP reference using LIPM
        """
        n_points = len(zmp_trajectory)
        com_trajectory = np.zeros((n_points, 3))
        com_velocity = np.zeros((n_points, 3))
        com_acceleration = np.zeros((n_points, 3))

        # Initial conditions
        com_trajectory[0, 2] = self.com_height  # Constant height

        # Integrate LIPM equations: COM_ddot = omega^2 * (COM - ZMP)
        for i in range(1, n_points):
            # Acceleration based on ZMP error
            zmp_error = com_trajectory[i-1, :2] - zmp_trajectory[i-1, :2]
            com_acceleration[i, :2] = self.omega**2 * zmp_error

            # Update velocity and position
            com_velocity[i, :2] = com_velocity[i-1, :2] + com_acceleration[i, :2] * dt
            com_trajectory[i, :2] = com_trajectory[i-1, :2] + com_velocity[i, :2] * dt

            # Keep height constant
            com_trajectory[i, 2] = self.com_height

        return com_trajectory, com_velocity, com_acceleration
```

## Preview Control for Walking

### Preview Control Theory

Preview control uses future reference information to improve tracking performance:

```python
class PreviewController:
    def __init__(self, com_height, preview_time=2.0, dt=0.01):
        self.com_height = com_height
        self.gravity = 9.81
        self.omega = np.sqrt(self.gravity / com_height)
        self.dt = dt
        self.preview_steps = int(preview_time / dt)

        # Calculate preview control gains
        self.Kx, self.Kr = self.calculate_preview_gains()

    def calculate_preview_gains(self):
        """
        Calculate preview control gains using Riccati equation solution
        """
        # System matrices for LIPM
        # State: [x, x_dot], Reference: zmp_ref
        A = np.array([[1, self.dt], [self.omega**2 * self.dt, 1]])
        B = np.array([0, -self.omega**2 * self.dt])

        # Cost matrices
        Q = np.array([[10, 0], [0, 1]])  # State cost
        R = np.array([[0.1]])            # Control cost

        # Solve discrete-time Riccati equation
        P = self.solve_dare(A, B, Q, R)

        # Feedback gain
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

        # Calculate preview gains
        # This is a simplified calculation - in practice, this involves
        # solving for the preview gain matrix
        Kx = K
        Kr = np.zeros((1, self.preview_steps))  # Simplified

        # Calculate actual preview gains (more complex in practice)
        for i in range(self.preview_steps):
            # Preview gain calculation would go here
            Kr[0, i] = self.calculate_preview_gain(i)

        return Kx, Kr

    def solve_dare(self, A, B, Q, R):
        """
        Solve discrete-time algebraic Riccati equation
        Simplified implementation
        """
        # This is a simplified version - in practice, use scipy.linalg.solve_discrete_are
        P = Q.copy()
        for _ in range(50):  # Iterative solution
            P_new = A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A + Q
            if np.allclose(P, P_new, rtol=1e-6):
                break
            P = P_new
        return P

    def calculate_preview_gain(self, k):
        """
        Calculate preview gain for step k
        """
        # Simplified preview gain calculation
        # In practice, this involves the solution of the Riccati equation
        # and the system's impulse response
        if k == 0:
            return 1.0
        else:
            return np.exp(-k * self.dt * self.omega)  # Exponential decay

    def compute_control(self, current_state, zmp_reference):
        """
        Compute control using preview information
        current_state: [x, x_dot] current CoM state
        zmp_reference: array of future ZMP references
        """
        # Feedback control
        feedback_control = -self.Kx @ current_state

        # Preview control
        preview_control = 0
        n_future = min(len(zmp_reference), self.preview_steps)

        for k in range(n_future):
            preview_control += self.Kr[0, k] * zmp_reference[k]

        total_control = feedback_control + preview_control
        return total_control
```

### Walking with Preview Control

```python
class PreviewWalkingController:
    def __init__(self, com_height, step_length=0.3, step_width=0.2, dt=0.01):
        self.com_height = com_height
        self.step_length = step_length
        self.step_width = step_width
        self.dt = dt

        self.preview_controller = PreviewController(com_height, dt=dt)
        self.zmp_planner = ZMPTrajectoryPlanner(com_height, step_length, step_width)

    def generate_walking_trajectory(self, footsteps, initial_com_pos, initial_com_vel):
        """
        Generate walking trajectory using preview control
        """
        # Plan ZMP trajectory
        zmp_ref = self.zmp_planner.plan_zmp_trajectory(footsteps, self.dt)

        # Initialize trajectories
        n_points = len(zmp_ref)
        com_trajectory = np.zeros((n_points, 3))
        com_velocity = np.zeros((n_points, 3))
        com_acceleration = np.zeros((n_points, 3))

        # Set initial conditions
        com_trajectory[0, :2] = initial_com_pos[:2]
        com_trajectory[0, 2] = self.com_height
        com_velocity[0, :2] = initial_com_vel[:2]

        # Generate trajectory using preview control
        for i in range(1, n_points):
            # Current state
            current_state = np.array([
                com_trajectory[i-1, 0],  # x position
                com_velocity[i-1, 0]     # x velocity
            ])

            # Future ZMP references (preview window)
            preview_end = min(i + self.preview_controller.preview_steps, n_points)
            zmp_future = zmp_ref[i:preview_end, 0]  # X component

            # Compute control (simplified - only showing X direction)
            control_x = self.preview_controller.compute_control(current_state, zmp_future)

            # Apply control to update state
            # This is a simplified integration of the LIPM equations
            com_acceleration[i, 0] = self.preview_controller.omega**2 * (
                com_trajectory[i-1, 0] - zmp_ref[i-1, 0]
            )
            com_velocity[i, 0] = com_velocity[i-1, 0] + com_acceleration[i, 0] * self.dt
            com_trajectory[i, 0] = com_trajectory[i-1, 0] + com_velocity[i, 0] * self.dt

            # Similar for Y direction
            current_state_y = np.array([
                com_trajectory[i-1, 1],  # y position
                com_velocity[i-1, 1]     # y velocity
            ])
            zmp_future_y = zmp_ref[i:preview_end, 1]  # Y component
            control_y = self.preview_controller.compute_control(current_state_y, zmp_future_y)

            com_acceleration[i, 1] = self.preview_controller.omega**2 * (
                com_trajectory[i-1, 1] - zmp_ref[i-1, 1]
            )
            com_velocity[i, 1] = com_velocity[i-1, 1] + com_acceleration[i, 1] * self.dt
            com_trajectory[i, 1] = com_trajectory[i-1, 1] + com_velocity[i, 1] * self.dt

            # Keep height constant
            com_trajectory[i, 2] = self.com_height

        return {
            'com_trajectory': com_trajectory,
            'com_velocity': com_velocity,
            'com_acceleration': com_acceleration,
            'zmp_reference': zmp_ref
        }
```

## Adaptive and Robust Walking

### Terrain Adaptation

```python
class AdaptiveWalkingController:
    def __init__(self, base_controller):
        self.base_controller = base_controller
        self.terrain_estimator = TerrainEstimator()
        self.adaptation_gains = {
            'step_height': 0.5,
            'step_length': 0.3,
            'walking_speed': 0.2
        }

    def adapt_to_terrain(self, terrain_info, base_footsteps):
        """
        Adapt walking pattern based on terrain information
        terrain_info: Dictionary with terrain properties
        """
        adapted_footsteps = []

        for step in base_footsteps:
            # Get terrain properties at step location
            terrain_at_step = self.terrain_estimator.get_terrain_properties(
                step['position'][:2]
            )

            # Adapt step parameters based on terrain
            adapted_step = step.copy()

            # Adjust step height for obstacles
            if terrain_at_step['obstacle_height'] > 0.05:  # 5cm threshold
                adapted_step['step_height'] = max(
                    0.1,  # Minimum clearance
                    terrain_at_step['obstacle_height'] + 0.05  # 5cm extra clearance
                )
            else:
                adapted_step['step_height'] = 0.05  # Normal step height

            # Adjust step length for slippery surfaces
            if terrain_at_step['friction_coeff'] < 0.4:  # Low friction
                adapted_step['step_length'] = min(
                    step['step_length'] * 0.8,  # Reduce by 20%
                    self.adaptation_gains['step_length']
                )

            # Adjust step width for unstable terrain
            if terrain_at_step['stability'] < 0.7:  # Unstable terrain
                adapted_step['step_width'] = max(
                    step['step_width'] * 1.2,  # Increase width by 20%
                    self.adaptation_gains['step_width']
                )

            adapted_footsteps.append(adapted_step)

        return adapted_footsteps

class TerrainEstimator:
    def __init__(self):
        self.terrain_map = {}  # Could be a grid map or other representation

    def get_terrain_properties(self, position):
        """
        Estimate terrain properties at given position
        """
        x, y = position

        # In practice, this would query a terrain map or use sensors
        # For simulation, return synthetic properties
        properties = {
            'obstacle_height': self.estimate_obstacle_height(x, y),
            'friction_coeff': self.estimate_friction(x, y),
            'stability': self.estimate_stability(x, y),
            'slope': self.estimate_slope(x, y)
        }

        return properties

    def estimate_obstacle_height(self, x, y):
        """Estimate obstacle height at position"""
        # In practice, this would use perception data
        return 0.0  # Assume flat terrain for now

    def estimate_friction(self, x, y):
        """Estimate friction coefficient at position"""
        # In practice, this would use terrain classification
        return 0.8  # Assume normal friction

    def estimate_stability(self, x, y):
        """Estimate terrain stability at position"""
        # In practice, this would use terrain analysis
        return 1.0  # Assume stable terrain

    def estimate_slope(self, x, y):
        """Estimate ground slope at position"""
        # In practice, this would use terrain normals
        return 0.0  # Assume flat terrain
```

### Disturbance Rejection

```python
class DisturbanceRejectionController:
    def __init__(self, base_controller):
        self.base_controller = base_controller
        self.disturbance_observer = DisturbanceObserver()
        self.recovery_controller = BalanceRecoveryController()

    def handle_disturbance(self, current_state, measured_zmp, reference_zmp):
        """
        Detect and handle external disturbances
        """
        # Calculate ZMP error (indicates disturbance)
        zmp_error = measured_zmp - reference_zmp

        # Check if disturbance exceeds threshold
        if np.linalg.norm(zmp_error) > 0.05:  # 5cm threshold
            # Estimate disturbance using observer
            estimated_disturbance = self.disturbance_observer.estimate(
                current_state, measured_zmp, reference_zmp
            )

            # Apply recovery control
            recovery_control = self.recovery_controller.compute_recovery(
                current_state, estimated_disturbance
            )

            return recovery_control
        else:
            # No significant disturbance, return normal control
            return self.base_controller.compute_normal_control(
                current_state, reference_zmp
            )

class DisturbanceObserver:
    def __init__(self):
        self.disturbance_estimate = np.zeros(2)  # x, y disturbances
        self.filter_gain = 0.1

    def estimate(self, current_state, measured_zmp, reference_zmp):
        """
        Estimate external disturbance using state observer
        """
        # Simple disturbance estimation
        # In practice, this would use more sophisticated observer design
        zmp_error = measured_zmp - reference_zmp

        # Update disturbance estimate using low-pass filter
        self.disturbance_estimate = (
            (1 - self.filter_gain) * self.disturbance_estimate +
            self.filter_gain * zmp_error
        )

        return self.disturbance_estimate

class BalanceRecoveryController:
    def __init__(self):
        self.critical_zmp_distance = 0.08  # 8cm from foot edge

    def compute_recovery(self, current_state, estimated_disturbance):
        """
        Compute recovery control to regain balance
        """
        # Calculate capture point
        com_pos = current_state[:2]
        com_vel = current_state[2:4]
        capture_point = self.calculate_capture_point(com_pos, com_vel)

        # Check if capture point is outside support polygon
        if self.is_capture_point_unstable(capture_point):
            # Execute recovery strategy
            recovery_control = self.execute_recovery_strategy(
                current_state, capture_point
            )
        else:
            # Apply disturbance compensation
            recovery_control = -estimated_disturbance * 0.5  # Damping

        return recovery_control

    def calculate_capture_point(self, com_pos, com_vel):
        """
        Calculate capture point for recovery
        """
        # Simplified capture point calculation
        com_height = 0.8  # Assumed CoM height
        gravity = 9.81
        omega = np.sqrt(gravity / com_height)

        capture_point_x = com_pos[0] + com_vel[0] / omega
        capture_point_y = com_pos[1] + com_vel[1] / omega

        return np.array([capture_point_x, capture_point_y])

    def is_capture_point_unstable(self, capture_point):
        """
        Check if capture point indicates instability
        """
        # Simplified check - in practice, compare with foot positions
        return np.linalg.norm(capture_point) > self.critical_zmp_distance

    def execute_recovery_strategy(self, current_state, capture_point):
        """
        Execute specific recovery strategy
        """
        # Determine appropriate recovery action
        if np.abs(capture_point[0]) > np.abs(capture_point[1]):
            # Forward/backward instability - adjust CoM height or step
            return self.forward_backward_recovery(current_state, capture_point)
        else:
            # Lateral instability - step sideways or adjust stance width
            return self.lateral_recovery(current_state, capture_point)

    def forward_backward_recovery(self, current_state, capture_point):
        """Recovery strategy for forward/backward instability"""
        # In practice, this might involve stepping, crouching, or arm movements
        recovery_control = -np.array([capture_point[0] * 0.3, 0])
        return recovery_control

    def lateral_recovery(self, current_state, capture_point):
        """Recovery strategy for lateral instability"""
        # In practice, this might involve stepping sideways
        recovery_control = -np.array([0, capture_point[1] * 0.3])
        return recovery_control
```

## Implementation Considerations

### Real-time Performance

For real-time walking control, performance optimization is crucial:

```python
import numba
from numba import jit

@jit(nopython=True)
def fast_zmp_calculation(com_pos, com_acc, com_height, gravity):
    """
    Fast ZMP calculation using Numba JIT compilation
    """
    zmp_x = com_pos[0] - (com_height * com_acc[0]) / gravity
    zmp_y = com_pos[1] - (com_height * com_acc[1]) / gravity

    return zmp_x, zmp_y

@jit(nopython=True)
def fast_com_integration(com_pos, com_vel, com_acc, dt):
    """
    Fast CoM state integration
    """
    new_vel_x = com_vel[0] + com_acc[0] * dt
    new_vel_y = com_vel[1] + com_acc[1] * dt
    new_pos_x = com_pos[0] + new_vel_x * dt
    new_pos_y = com_pos[1] + new_vel_y * dt

    return new_pos_x, new_pos_y, new_vel_x, new_vel_y

class OptimizedWalkingController:
    def __init__(self, com_height, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / com_height)
        self.dt = 0.01

    def update_step(self, current_com_pos, current_com_vel, zmp_reference):
        """
        Fast single step update using optimized calculations
        """
        # Fast ZMP calculation
        zmp_x, zmp_y = fast_zmp_calculation(
            current_com_pos, current_com_vel, self.com_height, self.gravity
        )

        # Fast CoM acceleration calculation (LIPM)
        acc_x = self.omega**2 * (current_com_pos[0] - zmp_x)
        acc_y = self.omega**2 * (current_com_pos[1] - zmp_y)

        # Fast integration
        new_pos_x, new_pos_y, new_vel_x, new_vel_y = fast_com_integration(
            current_com_pos, current_com_vel, np.array([acc_x, acc_y]), self.dt
        )

        return (new_pos_x, new_pos_y), (new_vel_x, new_vel_y), (acc_x, acc_y)
```

### Safety and Validation

Humanoid walking requires extensive safety measures:

```python
class WalkingSafetyValidator:
    def __init__(self, robot_limits):
        self.joint_limits = robot_limits['joint_limits']
        self.torque_limits = robot_limits['torque_limits']
        self.velocity_limits = robot_limits['velocity_limits']

    def validate_walking_pattern(self, walking_pattern):
        """
        Validate walking pattern for safety
        """
        issues = []

        # Check joint limit violations
        joint_angles = walking_pattern.get('joint_angles', [])
        for i, angles in enumerate(joint_angles):
            for j, angle in enumerate(angles):
                min_limit, max_limit = self.joint_limits[j]
                if angle < min_limit or angle > max_limit:
                    issues.append(f"Joint limit violation at step {i}, joint {j}: {angle:.3f}")

        # Check torque limit violations
        torques = walking_pattern.get('torques', [])
        for i, torque_set in enumerate(torques):
            for j, torque in enumerate(torque_set):
                if abs(torque) > self.torque_limits[j]:
                    issues.append(f"Torque limit violation at step {i}, joint {j}: {torque:.3f}")

        # Check for stability
        com_trajectory = walking_pattern.get('com_trajectory', [])
        zmp_trajectory = walking_pattern.get('zmp_trajectory', [])

        if len(com_trajectory) > 0 and len(zmp_trajectory) > 0:
            stability_checker = ZMPController(com_height=0.8)
            support_polygon = self.calculate_support_polygon(walking_pattern)

            for i, (com_pos, zmp_pos) in enumerate(zip(com_trajectory, zmp_trajectory)):
                if not stability_checker.is_stable(zmp_pos, support_polygon):
                    issues.append(f"Stability violation at time step {i}")

        return len(issues) == 0, issues

    def calculate_support_polygon(self, walking_pattern):
        """
        Calculate support polygon based on foot positions
        """
        # Simplified rectangular support polygon
        # In practice, this would use actual foot positions
        foot_positions = walking_pattern.get('foot_positions', [])

        if len(foot_positions) == 0:
            # Default support polygon
            return [
                (-0.1, -0.05),  # back left
                (0.1, -0.05),   # front left
                (0.1, 0.05),    # front right
                (-0.1, 0.05)    # back right
            ]

        # Calculate from actual foot positions
        # This is a simplified example
        return [
            (-0.1, -0.1),
            (0.2, -0.1),
            (0.2, 0.1),
            (-0.1, 0.1)
        ]

    def emergency_stop(self, current_state):
        """
        Execute emergency stop procedure
        """
        # Calculate safe stopping trajectory
        capture_point = self.calculate_capture_point(current_state)

        # Plan immediate step to capture point
        emergency_step = {
            'position': (capture_point[0], capture_point[1], 0),
            'type': 'emergency_stop'
        }

        return emergency_step

    def calculate_capture_point(self, current_state):
        """
        Calculate capture point for emergency stopping
        """
        com_pos = current_state['com_position'][:2]
        com_vel = current_state['com_velocity'][:2]
        com_height = current_state['com_height']

        gravity = 9.81
        omega = np.sqrt(gravity / com_height)

        capture_x = com_pos[0] + com_vel[0] / omega
        capture_y = com_pos[1] + com_vel[1] / omega

        return np.array([capture_x, capture_y])
```

## Exercises

1. Implement a simple inverted pendulum walking controller and test its stability with different step lengths.

2. Generate a walking pattern for a 10-step walk and visualize the CoM and ZMP trajectories.

3. Implement a balance controller that can recover from small external disturbances.

4. Create a terrain adaptation system that modifies walking parameters based on surface properties.

5. Implement preview control for walking and compare its performance with non-preview control.

6. Design a walking controller that can handle stairs or uneven terrain.

7. Implement a disturbance observer to estimate external forces during walking.

8. Create a walking pattern that transitions smoothly between different walking speeds.

9. Implement a capture point-based recovery strategy for when balance is lost.

10. Validate your walking controller with a physics simulation and measure stability margins.

## Next Steps

After completing this chapter, you should have a comprehensive understanding of bipedal locomotion principles and implementation. The next chapter will explore manipulation and grasping techniques for humanoid robots, building on the locomotion capabilities to enable full-body interaction with the environment.