# Clarity Document: Physical AI Robotics Book

## Purpose
A comprehensive, student-friendly, technically accurate book for Quarter 4: Physical AI & Humanoid Robotics, created using Docusaurus and deployed on GitHub Pages. The book will bridge AI knowledge (LLMs, VLA, perception), robotics fundamentals (ROS 2, URDF, SLAM), simulation (Isaac, Gazebo, Unity), humanoid robot mechanics, and real-world deployment (Jetson + sensors + robotic platforms).

## Goals
The book aims to enable students learning Physical AI for the first time, developers transitioning from software/AI to robotics, and Hackathon participants using Spec-Kit Plus to apply AI knowledge to control Humanoid Robots in simulated and real-world environments. It will assume zero prior experience with robotics.

## Audience
- Students learning Physical AI for the first time
- Developers transitioning from software/AI to robotics
- Hackathon participants using Spec-Kit Plus
- Readers with basic Python/AI knowledge but little robotics experience

## Book Structure

### Chapter-by-chapter Outline

#### PART 1 — Foundations of Physical AI
-   **Chapter 1: What is Physical AI?**
    -   Digital vs Physical AI
    -   Embodied intelligence
    -   Why humanoid robots matter
    -   How AI interacts with physics
    -   Modern robotics ecosystem
-   **Chapter 2: The Humanoid Landscape**
    -   Unitree, Agility, Tesla, Boston Dynamics
    -   Sensors: LiDAR, IMU, Cameras
    -   Actuators
    -   Common robot stacks
-   **Chapter 3: Course Overview**
    -   Modules 1–4
    -   Capstone explanation
    -   Weekly roadmap summary

#### PART 2 — Module 1: The Robotic Nervous System (ROS 2)
-   **Chapter 4: ROS 2 Fundamentals**
    -   Nodes, Topics, Services, Actions
    -   DDS
    -   How ROS 2 abstracts hardware
-   **Chapter 5: Building ROS Packages**
    -   Python-based ROS packages
    -   Launch files
    -   Parameterization
-   **Chapter 6: URDF for Humanoids**
    -   Joints, links, transmissions
    -   Building humanoid models
    -   Understanding kinematics

#### PART 3 — Module 2: Digital Twin (Gazebo + Unity)
-   **Chapter 7: Gazebo Physics Engine**
    -   Gravity, friction, collisions
    -   Sensors simulation
-   **Chapter 8: Unity for Robotics**
    -   Visual rendering
    -   Human-robot interaction simulation

#### PART 4 — Module 3: The AI-Robot Brain (NVIDIA Isaac)
-   **Chapter 9: Isaac Sim**
    -   USD worlds
    -   Synthetic data
    -   Perception and manipulation
-   **Chapter 10: Isaac ROS**
    -   VSLAM
    -   Navigation (Nav2)
    -   GPU-accelerated perception

#### PART 5 — Module 4: Vision-Language-Action (VLA)
-   **Chapter 11: Convergence of LLMs + Robotics**
    -   Using LLMs for planning
    -   Multimodal perception
-   **Chapter 12: Voice-to-Action**
    -   Whisper → ROS pipeline
    -   Natural language → robotic action steps

#### PART 6 — Capstone Project: The Autonomous Humanoid
-   **Chapter 13: Project Specification**
    -   Voice command
    -   Path planning
    -   Navigation
    -   Obstacle avoidance
    -   Object detection
    -   Grasping + manipulation
-   **Chapter 14: Evaluation Rubric**
    -   Simulation
    -   Perception
    -   Planning
    -   VLA integration

#### PART 7 — Hardware Requirements & Lab Architecture
-   **Chapter 15: Workstation Requirements**
    -   GPU / RAM / CPU requirements
    -   Linux setup
-   **Chapter 16: Jetson Edge Kit**
    -   Sensors
    -   IMU
    -   RealSense Camera
-   **Chapter 17: Robot Lab Options**
    -   Unitree Go2
    -   Unitree G1
    -   Hiwonder kits
    -   Proxy robots

#### PART 8 — Weekly Breakdown
-   Each week contains:
    -   Topic overview
    -   Learning goals
    -   Assignments

#### PART 9 — Assessments & Final Evaluation
-   ROS package project
-   Gazebo simulation project
-   Isaac perception pipeline
-   Capstone project grading

## Scope Boundaries

### The book MUST include:
- Physical AI foundational concepts
- Embodied intelligence
- ROS 2 architecture
- URDF & robot description
- Gazebo physics simulation
- Unity rendering for humanoid visualization
- NVIDIA Isaac Sim + Isaac ROS + Nav2
- VLA systems (Vision-Language-Action)
- OpenAI Whisper → ROS integration
- Capstone: Autonomous Humanoid Robot
- Hardware recommendations
- Weekly syllabus
- Assessment breakdown
- High-performance workstation requirements
- Jetson Edge Kit details

### The book MUST NOT include:
- Deep mathematics proofs
- ROS 1
- Full source code of ROS packages
- GPU installation guides
- Cloud setup tutorials (only high-level overview allowed)

## Tools
- Spec-Kit Plus for book creation workflow
- Claude Code for AI-powered writing
- Docusaurus for book rendering
- GitHub Pages for deployment
- Markdown as the core writing format
- Diagrams generated via Mermaid.js where needed

## Folder Architecture
`/docs`
  `/module-1-ros2`
  `/module-2-digital-twin`
  `/module-3-isaac`
  `/module-4-vla`
  `/capstone`
  `/hardware-requirements`
  `/foundations`
  `/weekly-breakdown`
  `/assessments`
  `/lab-architecture`
  `/economy-kit`

## Tone & Style Guidelines
- Educational, practical, readable, visually clear, project-oriented.
- Explanations should be simple.
- Use diagrams.
- Follow progressive complexity.
- Avoid long code dumps.
- Use headings, subheadings, and bullets.
- Include summaries at the end of each chapter.

## Expected Agent Behavior
- The agent must follow the fixed book structure.
- Each page must include: Title, Overview, Key learning outcomes, Clear diagrams (text-based if needed), Step-by-step explanations, Minimal code examples, Glossary section if technical terms appear.
- The agent must adhere to the content boundaries (MUST include/MUST NOT include).
- The agent must follow the specified flow of the book (narrative order of parts and chapters).
- The agent must adhere to the writing constraints (simple explanations, diagrams, progressive complexity, no long code dumps, headings/subheadings/bullets, chapter summaries).
