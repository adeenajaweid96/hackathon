# LLM Cognitive Planning for Humanoid Robots

## Overview
This chapter explores the integration of Large Language Models (LLMs) for cognitive planning in humanoid robots. LLMs serve as the "brain" of the robot, processing natural language commands and generating high-level plans that can be executed by the robot's action system.

## Learning Objectives
- Understand the role of LLMs in robotic cognitive planning
- Learn to integrate popular LLMs (OpenAI GPT, Anthropic Claude, etc.) with ROS 2
- Design prompt engineering techniques for robotic applications
- Create action planning systems that translate LLM outputs to robot commands
- Implement safety and validation mechanisms for LLM-generated plans
- Evaluate and optimize LLM performance for real-time robotic applications

## Prerequisites
- Understanding of ROS 2 concepts and message types
- Basic knowledge of natural language processing
- Completed Whisper integration chapter
- Familiarity with robot action execution systems

## Table of Contents
1. [Introduction to LLMs in Robotics](#introduction-to-llms-in-robotics)
2. [LLM Integration Options](#llm-integration-options)
3. [Prompt Engineering for Robotics](#prompt-engineering-for-robotics)
4. [Action Planning and Execution](#action-planning-and-execution)
5. [ROS 2 Integration](#ros-2-integration)
6. [Safety and Validation](#safety-and-validation)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting Common Issues](#troubleshooting-common-issues)
9. [Best Practices](#best-practices)

## Introduction to LLMs in Robotics

### The Cognitive Layer
LLMs serve as the cognitive layer in humanoid robots, bridging the gap between natural language input and robotic action execution. This cognitive layer performs several key functions:

1. **Language Understanding**: Interpret natural language commands
2. **Task Decomposition**: Break complex tasks into executable actions
3. **Context Management**: Maintain context across interactions
4. **Reasoning**: Apply logical reasoning to solve problems
5. **Learning**: Adapt to user preferences and environmental changes

### Benefits of LLM Integration
- **Natural Interaction**: Enable human-like communication with robots
- **Flexibility**: Handle diverse and novel commands
- **Adaptability**: Learn from interactions and improve over time
- **Scalability**: Leverage pre-trained knowledge from large datasets

### Challenges in Robotic Applications
- **Real-time Constraints**: Balancing quality with response time
- **Safety**: Ensuring safe execution of LLM-generated plans
- **Reliability**: Handling uncertain or ambiguous commands
- **Embodiment**: Translating abstract language to physical actions

## LLM Integration Options

### OpenAI GPT Integration
OpenAI's GPT models are popular choices for robotic applications due to their strong language understanding capabilities:

```python
import openai
import asyncio
from typing import Dict, List, Optional

class GPTPlanner:
    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        openai.api_key = api_key
        self.model = model
        self.conversation_history = []

    async def generate_plan(self, user_command: str, robot_state: Dict) -> Dict:
        """Generate a plan based on user command and robot state"""
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(user_command, robot_state)

        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        plan_text = response.choices[0].message.content
        return self._parse_plan(plan_text)

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM"""
        return """
        You are an AI cognitive planner for a humanoid robot. Your role is to interpret natural language commands and generate detailed action plans that the robot can execute.

        Guidelines:
        1. Always consider the robot's current state and capabilities
        2. Break down complex commands into simple, executable actions
        3. Include safety checks and validation steps
        4. Provide clear, step-by-step instructions
        5. Handle ambiguous requests by asking for clarification
        6. Prioritize safety in all generated plans

        Available robot capabilities:
        - Navigation: Move to specific locations
        - Manipulation: Pick up, place, and manipulate objects
        - Perception: Detect and recognize objects and people
        - Communication: Speak, gesture, and display information

        Respond in JSON format with the following structure:
        {
          "plan": [
            {
              "action": "action_type",
              "parameters": {"param1": "value1", ...},
              "description": "Human-readable description"
            }
          ],
          "confidence": 0.0-1.0,
          "reasoning": "Brief explanation of your plan"
        }
        """

    def _build_user_prompt(self, command: str, robot_state: Dict) -> str:
        """Build the user prompt with context"""
        return f"""
        Robot State: {robot_state}
        User Command: {command}

        Generate a detailed action plan to fulfill the user's request.
        """

    def _parse_plan(self, plan_text: str) -> Dict:
        """Parse the LLM response into a structured plan"""
        import json
        import re

        # Extract JSON from the response
        json_match = re.search(r'\{.*\}', plan_text, re.DOTALL)
        if json_match:
            try:
                plan = json.loads(json_match.group())
                return plan
            except json.JSONDecodeError:
                pass

        # If JSON parsing fails, return a basic structure
        return {
            "plan": [{"action": "speak", "parameters": {"text": plan_text}, "description": "Speak response"}],
            "confidence": 0.5,
            "reasoning": "Failed to parse structured response"
        }
```

### Anthropic Claude Integration
Anthropic's Claude models offer strong reasoning capabilities and safety features:

```python
import anthropic
import json
from typing import Dict, Any

class ClaudePlanner:
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate_plan(self, user_command: str, robot_state: Dict) -> Dict:
        """Generate a plan using Claude"""
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(user_command, robot_state)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0.3,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        plan_text = response.content[0].text
        return self._parse_plan(plan_text)

    def _build_system_prompt(self) -> str:
        """Build system prompt for Claude"""
        return """
        You are an AI cognitive planner for a humanoid robot. Your role is to interpret natural language commands and generate detailed action plans that the robot can execute safely.

        Guidelines:
        1. Always consider the robot's current state and physical capabilities
        2. Break complex tasks into simple, executable actions
        3. Include safety checks and validation for each action
        4. Provide clear, step-by-step instructions in JSON format
        5. Handle ambiguous requests by asking for clarification
        6. Prioritize safety and feasibility in all generated plans

        Available robot capabilities:
        - Navigation: Move to specific locations (navigate_to(location))
        - Manipulation: Pick up, place, and manipulate objects (pick_up(object), place_object(object, location))
        - Perception: Detect and recognize objects and people (detect_object(object))
        - Communication: Speak, gesture, and display information (speak(text), gesture(gesture_type))

        Respond in JSON format with the following structure:
        {
          "plan": [
            {
              "action": "action_type",
              "parameters": {"param1": "value1", ...},
              "description": "Human-readable description",
              "safety_check": "description of safety validation needed"
            }
          ],
          "confidence": 0.0-1.0,
          "reasoning": "Brief explanation of your plan and why it will work"
        }
        """

    def _build_user_prompt(self, command: str, robot_state: Dict) -> str:
        """Build user prompt with context"""
        return f"""
        Current Robot State: {json.dumps(robot_state, indent=2)}

        User Command: {command}

        Generate a detailed action plan to fulfill the user's request. Ensure all actions are safe and executable given the robot's current state.
        """

    def _parse_plan(self, plan_text: str) -> Dict:
        """Parse Claude's response into structured plan"""
        import re
        import json

        # Extract JSON from Claude's response
        json_match = re.search(r'```json\n(.*?)\n```', plan_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{.*\}', plan_text, re.DOTALL)

        if json_match:
            try:
                json_str = json_match.group()
                if json_str.startswith('```json'):
                    json_str = json_match.group(1)
                plan = json.loads(json_str)
                return plan
            except json.JSONDecodeError:
                pass

        # Fallback: return a basic plan
        return {
            "plan": [{"action": "speak", "parameters": {"text": f"I understand: {plan_text}"}, "description": "Acknowledge user command", "safety_check": "None"}],
            "confidence": 0.3,
            "reasoning": "Failed to parse structured response from LLM"
        }
```

### Open-Source LLM Options
For privacy-conscious or offline applications, consider open-source models:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class OpenSourcePlanner:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_plan(self, user_command: str, robot_state: Dict) -> Dict:
        """Generate plan using open-source model"""
        prompt = self._build_prompt(user_command, robot_state)

        inputs = self.tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 200,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_response(response)

    def _build_prompt(self, command: str, robot_state: Dict) -> str:
        """Build prompt for open-source model"""
        return f"""
        Robot State: {robot_state}
        User Command: {command}
        Action Plan:
        """
```

## Prompt Engineering for Robotics

### Context-Aware Prompting
Provide rich context to improve LLM performance:

```python
class ContextualPromptBuilder:
    def __init__(self):
        self.system_context = self._get_system_context()
        self.environment_context = {}
        self.user_preferences = {}

    def build_cognitive_prompt(self, user_command: str, robot_state: Dict,
                              environment_data: Dict = None,
                              user_history: List = None) -> str:
        """Build a comprehensive prompt with all relevant context"""

        # Environment context
        env_context = self._format_environment_context(environment_data)

        # User history context
        history_context = self._format_user_history(user_history)

        # Robot capabilities context
        capabilities_context = self._format_robot_capabilities(robot_state)

        # Safety constraints
        safety_context = self._get_safety_constraints()

        prompt = f"""
        SYSTEM CONTEXT:
        {self.system_context}

        SAFETY CONSTRAINTS:
        {safety_context}

        ROBOT CAPABILITIES:
        {capabilities_context}

        ENVIRONMENT STATE:
        {env_context}

        USER INTERACTION HISTORY:
        {history_context}

        CURRENT ROBOT STATE:
        {robot_state}

        USER COMMAND:
        {user_command}

        Generate a detailed action plan that:
        1. Achieves the user's goal safely
        2. Respects all safety constraints
        3. Uses available robot capabilities effectively
        4. Considers environmental constraints
        5. Maintains user preferences and history

        Respond in JSON format with the action plan.
        """

        return prompt

    def _get_system_context(self) -> str:
        """Get general system context"""
        return """
        You are controlling a humanoid robot in a real-world environment.
        The robot has physical constraints and safety requirements.
        Always prioritize safety over task completion.
        """

    def _format_environment_context(self, env_data: Dict) -> str:
        """Format environment data for prompt"""
        if not env_data:
            return "Environment: Unknown or not provided"

        formatted = []
        for key, value in env_data.items():
            formatted.append(f"{key}: {value}")

        return "\n".join(formatted)

    def _format_user_history(self, history: List) -> str:
        """Format user interaction history"""
        if not history:
            return "No previous interactions"

        formatted = []
        for i, interaction in enumerate(history[-5:], 1):  # Last 5 interactions
            formatted.append(f"Interaction {i}: {interaction}")

        return "\n".join(formatted)

    def _format_robot_capabilities(self, state: Dict) -> str:
        """Format robot capabilities and current state"""
        capabilities = [
            "Navigation: Can move to locations",
            "Manipulation: Can pick/place objects",
            "Perception: Can detect objects/people",
            "Communication: Can speak/gesture"
        ]

        return "\n".join(capabilities)

    def _get_safety_constraints(self) -> str:
        """Get safety constraints for the robot"""
        return """
        Safety Rules:
        1. Do not navigate to dangerous areas
        2. Do not manipulate objects that could cause harm
        3. Do not ignore obstacle detection
        4. Stop if safety sensors are triggered
        5. Always maintain safe distances from humans
        """
```

### Few-Shot Learning Examples
Provide examples to guide the LLM's behavior:

```python
FEW_SHOT_EXAMPLES = [
    {
        "input": "Please bring me a cup of coffee from the kitchen",
        "output": {
            "plan": [
                {
                    "action": "navigate_to",
                    "parameters": {"location": "kitchen"},
                    "description": "Move to the kitchen area",
                    "safety_check": "Check for obstacles along the path"
                },
                {
                    "action": "detect_object",
                    "parameters": {"object": "coffee_cup"},
                    "description": "Locate a coffee cup",
                    "safety_check": "Verify object is safe to handle"
                },
                {
                    "action": "pick_up",
                    "parameters": {"object": "coffee_cup"},
                    "description": "Grasp the coffee cup",
                    "safety_check": "Ensure proper grip and balance"
                },
                {
                    "action": "navigate_to",
                    "parameters": {"location": "user_location"},
                    "description": "Return to user location",
                    "safety_check": "Check for obstacles along the path"
                },
                {
                    "action": "place_object",
                    "parameters": {"object": "coffee_cup", "location": "delivery_position"},
                    "description": "Place the coffee cup near the user",
                    "safety_check": "Ensure safe placement location"
                }
            ],
            "confidence": 0.9,
            "reasoning": "User requested coffee delivery. Robot will navigate to kitchen, find a cup, pick it up, and deliver it to the user."
        }
    },
    {
        "input": "What time is it?",
        "output": {
            "plan": [
                {
                    "action": "get_time",
                    "parameters": {},
                    "description": "Retrieve current time",
                    "safety_check": "None required"
                },
                {
                    "action": "speak",
                    "parameters": {"text": "The current time is [time]"},
                    "description": "Announce the time to the user",
                    "safety_check": "None required"
                }
            ],
            "confidence": 1.0,
            "reasoning": "User asked for the time. Robot will retrieve current time and speak it aloud."
        }
    }
]

def add_few_shot_examples_to_prompt(base_prompt: str) -> str:
    """Add few-shot examples to the base prompt"""
    examples_text = "EXAMPLES:\n"
    for i, example in enumerate(FEW_SHOT_EXAMPLES, 1):
        examples_text += f"\nExample {i}:\n"
        examples_text += f"Input: {example['input']}\n"
        examples_text += f"Output: {json.dumps(example['output'], indent=2)}\n"

    return base_prompt + "\n" + examples_text
```

## Action Planning and Execution

### Plan Representation
Define a structured format for robot action plans:

```python
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum

class ActionType(Enum):
    NAVIGATE_TO = "navigate_to"
    PICK_UP = "pick_up"
    PLACE_OBJECT = "place_object"
    DETECT_OBJECT = "detect_object"
    SPEAK = "speak"
    GESTURE = "gesture"
    WAIT = "wait"
    GET_TIME = "get_time"
    CUSTOM_ACTION = "custom_action"

@dataclass
class ActionStep:
    action: ActionType
    parameters: Dict[str, Any]
    description: str
    safety_check: str
    expected_duration: float = 0.0  # in seconds
    timeout: float = 30.0  # in seconds

@dataclass
class ActionPlan:
    steps: List[ActionStep]
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]

class PlanExecutor:
    def __init__(self):
        self.current_plan: Optional[ActionPlan] = None
        self.current_step_index = 0
        self.execution_log = []

    async def execute_plan(self, plan: ActionPlan) -> Dict[str, Any]:
        """Execute the given action plan"""
        self.current_plan = plan
        self.current_step_index = 0
        self.execution_log = []

        for i, step in enumerate(plan.steps):
            self.current_step_index = i

            # Log execution
            self.execution_log.append({
                "step": i,
                "action": step.action.value,
                "status": "started",
                "timestamp": time.time()
            })

            # Perform safety check
            if not await self._perform_safety_check(step):
                return {
                    "success": False,
                    "error": f"Safety check failed for step {i}: {step.safety_check}",
                    "completed_steps": i,
                    "execution_log": self.execution_log
                }

            # Execute the action
            try:
                result = await self._execute_action_step(step)
                self.execution_log[-1].update({
                    "status": "completed",
                    "result": result,
                    "timestamp": time.time()
                })
            except Exception as e:
                self.execution_log[-1].update({
                    "status": "failed",
                    "error": str(e),
                    "timestamp": time.time()
                })
                return {
                    "success": False,
                    "error": f"Action failed at step {i}: {str(e)}",
                    "completed_steps": i,
                    "execution_log": self.execution_log
                }

        return {
            "success": True,
            "completed_steps": len(plan.steps),
            "execution_log": self.execution_log
        }

    async def _perform_safety_check(self, step: ActionStep) -> bool:
        """Perform safety check for the action step"""
        # This would interface with the robot's safety systems
        # For now, we'll return True for all steps
        print(f"Performing safety check: {step.safety_check}")
        return True

    async def _execute_action_step(self, step: ActionStep) -> Dict[str, Any]:
        """Execute a single action step"""
        # This would interface with ROS 2 action servers
        # For now, we'll simulate execution

        action_handlers = {
            ActionType.NAVIGATE_TO: self._handle_navigate_to,
            ActionType.PICK_UP: self._handle_pick_up,
            ActionType.PLACE_OBJECT: self._handle_place_object,
            ActionType.DETECT_OBJECT: self._handle_detect_object,
            ActionType.SPEAK: self._handle_speak,
            ActionType.GESTURE: self._handle_gesture,
            ActionType.WAIT: self._handle_wait,
            ActionType.GET_TIME: self._handle_get_time,
            ActionType.CUSTOM_ACTION: self._handle_custom_action
        }

        handler = action_handlers.get(step.action)
        if handler:
            return await handler(step.parameters)
        else:
            raise ValueError(f"Unknown action type: {step.action}")

    async def _handle_navigate_to(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle navigation action"""
        location = params.get("location", "unknown")
        print(f"Navigating to {location}")
        # Simulate navigation delay
        await asyncio.sleep(2)
        return {"status": "navigated", "location": location}

    async def _handle_pick_up(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pick up action"""
        obj = params.get("object", "unknown")
        print(f"Picking up {obj}")
        # Simulate pick up delay
        await asyncio.sleep(3)
        return {"status": "picked_up", "object": obj}

    # Additional handlers would be implemented similarly...
```

### Plan Validation and Refinement
Validate and refine LLM-generated plans before execution:

```python
class PlanValidator:
    def __init__(self):
        self.known_locations = set()
        self.known_objects = set()
        self.robot_capabilities = set()

    def validate_plan(self, plan: ActionPlan, robot_state: Dict) -> Dict[str, Any]:
        """Validate the action plan for safety and feasibility"""
        issues = []

        # Check each action step
        for i, step in enumerate(plan.steps):
            step_issues = self._validate_action_step(step, robot_state, i)
            issues.extend(step_issues)

        # Check overall plan structure
        structure_issues = self._validate_plan_structure(plan)
        issues.extend(structure_issues)

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "confidence_adjustment": self._calculate_confidence_adjustment(issues)
        }

    def _validate_action_step(self, step: ActionStep, robot_state: Dict, step_index: int) -> List[str]:
        """Validate a single action step"""
        issues = []

        # Check if action is supported by robot
        if step.action.value not in self.robot_capabilities:
            issues.append(f"Step {step_index}: Robot does not support action {step.action.value}")

        # Validate parameters for specific action types
        if step.action == ActionType.NAVIGATE_TO:
            location = step.parameters.get("location")
            if location and location not in self.known_locations:
                issues.append(f"Step {step_index}: Unknown location {location}")

        elif step.action == ActionType.PICK_UP:
            obj = step.parameters.get("object")
            if obj and obj not in self.known_objects:
                issues.append(f"Step {step_index}: Unknown object {obj}")

        elif step.action == ActionType.SPEAK:
            text = step.parameters.get("text", "")
            if not text.strip():
                issues.append(f"Step {step_index}: Empty speech text")

        return issues

    def _validate_plan_structure(self, plan: ActionPlan) -> List[str]:
        """Validate the overall plan structure"""
        issues = []

        # Check confidence level
        if plan.confidence < 0.5:
            issues.append("Plan confidence is too low (< 0.5)")

        # Check for empty plan
        if not plan.steps:
            issues.append("Plan contains no steps")

        # Check for circular navigation (simplified)
        navigate_steps = [step for step in plan.steps if step.action == ActionType.NAVIGATE_TO]
        if len(navigate_steps) > 10:
            issues.append("Plan contains too many navigation steps (possible loop)")

        return issues

    def _calculate_confidence_adjustment(self, issues: List[str]) -> float:
        """Calculate confidence adjustment based on issues found"""
        if not issues:
            return 0.0  # No adjustment needed

        # More issues = lower confidence
        severity_map = {
            "critical": -0.3,
            "high": -0.2,
            "medium": -0.1,
            "low": -0.05
        }

        adjustment = 0.0
        for issue in issues:
            # Simple categorization (in practice, you'd have more sophisticated logic)
            if "unknown location" in issue.lower():
                adjustment += severity_map["high"]
            elif "unknown object" in issue.lower():
                adjustment += severity_map["medium"]
            elif "empty" in issue.lower():
                adjustment += severity_map["low"]
            else:
                adjustment += severity_map["low"]

        return max(adjustment, -0.5)  # Don't reduce confidence below 0.5
```

## ROS 2 Integration

### Cognitive Planning Node
Create a ROS 2 node that integrates LLM planning with robot execution:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String
from vision_language_action_msgs.msg import CognitivePlan, PlanStep, PlanResult
from vision_language_action_msgs.srv import ExecutePlan
import json
import asyncio
from typing import Dict, Any

class CognitivePlannerNode(Node):
    def __init__(self):
        super().__init__('cognitive_planner_node')

        # Initialize LLM planner (using GPT as an example)
        api_key = self.declare_parameter('openai_api_key', '').value
        if not api_key:
            self.get_logger().error("OpenAI API key not provided")
            return

        self.llm_planner = GPTPlanner(api_key)

        # Publishers
        self.plan_pub = self.create_publisher(CognitivePlan, 'generated_plan', 10)
        self.result_pub = self.create_publisher(PlanResult, 'plan_result', 10)

        # Subscriber for commands
        self.command_sub = self.create_subscription(
            String,
            'voice_command',
            self.command_callback,
            QoSProfile(depth=10)
        )

        # Service for executing plans
        self.execute_service = self.create_service(
            ExecutePlan,
            'execute_plan',
            self.execute_plan_callback
        )

        # Plan executor
        self.plan_executor = PlanExecutor()

        self.get_logger().info("Cognitive planner node initialized")

    def command_callback(self, msg: String):
        """Callback for voice commands"""
        command = msg.data
        self.get_logger().info(f"Received command: {command}")

        # Get current robot state (this would come from robot state publisher)
        robot_state = self._get_robot_state()

        # Generate plan using LLM
        future = asyncio.run(self.llm_planner.generate_plan(command, robot_state))

        # Validate the plan
        validator = PlanValidator()
        validation_result = validator.validate_plan(future, robot_state)

        if validation_result["valid"]:
            # Convert to ROS 2 message
            plan_msg = self._convert_plan_to_ros_msg(future)
            self.plan_pub.publish(plan_msg)

            self.get_logger().info("Plan generated and published successfully")
        else:
            self.get_logger().error(f"Plan validation failed: {validation_result['issues']}")

    def execute_plan_callback(self, request: ExecutePlan.Request, response: ExecutePlan.Response):
        """Service callback for executing plans"""
        try:
            # Convert ROS message to internal format
            plan = self._convert_ros_plan_to_internal(request.plan)

            # Execute the plan
            execution_result = asyncio.run(self.plan_executor.execute_plan(plan))

            # Set response
            response.success = execution_result["success"]
            response.completed_steps = execution_result["completed_steps"]
            response.message = "Plan execution completed" if response.success else execution_result.get("error", "Unknown error")

            # Publish result
            result_msg = PlanResult()
            result_msg.success = response.success
            result_msg.completed_steps = response.completed_steps
            result_msg.message = response.message
            self.result_pub.publish(result_msg)

        except Exception as e:
            self.get_logger().error(f"Plan execution failed: {e}")
            response.success = False
            response.message = str(e)

        return response

    def _get_robot_state(self) -> Dict[str, Any]:
        """Get current robot state"""
        # This would interface with robot state publisher
        # For now, return a mock state
        return {
            "location": "starting_position",
            "battery_level": 0.85,
            "gripper_status": "open",
            "current_task": "idle",
            "available_actions": ["navigate", "speak", "detect", "manipulate"]
        }

    def _convert_plan_to_ros_msg(self, plan) -> CognitivePlan:
        """Convert internal plan format to ROS message"""
        plan_msg = CognitivePlan()
        plan_msg.confidence = plan.confidence
        plan_msg.reasoning = plan.reasoning

        for step in plan.steps:
            step_msg = PlanStep()
            step_msg.action = step.action.value
            step_msg.parameters = json.dumps(step.parameters)
            step_msg.description = step.description
            step_msg.safety_check = step.safety_check
            step_msg.expected_duration = step.expected_duration
            step_msg.timeout = step.timeout

            plan_msg.steps.append(step_msg)

        return plan_msg

    def _convert_ros_plan_to_internal(self, ros_plan) -> ActionPlan:
        """Convert ROS plan message to internal format"""
        steps = []
        for ros_step in ros_plan.steps:
            step = ActionStep(
                action=ActionType(ros_step.action),
                parameters=json.loads(ros_step.parameters),
                description=ros_step.description,
                safety_check=ros_step.safety_check,
                expected_duration=ros_step.expected_duration,
                timeout=ros_step.timeout
            )
            steps.append(step)

        return ActionPlan(
            steps=steps,
            confidence=ros_plan.confidence,
            reasoning=ros_plan.reasoning,
            metadata={}  # Add any additional metadata
        )

def main(args=None):
    rclpy.init(args=args)
    node = CognitivePlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Safety and Validation

### Safety Framework
Implement comprehensive safety measures for LLM-generated plans:

```python
class SafetyFramework:
    def __init__(self):
        self.safety_rules = self._load_safety_rules()
        self.environment_constraints = {}
        self.robot_constraints = {}

    def _load_safety_rules(self) -> Dict[str, Any]:
        """Load safety rules and constraints"""
        return {
            "navigation": {
                "min_distance_to_human": 0.5,  # meters
                "forbidden_areas": ["exit", "restricted", "construction"],
                "max_speed": 0.5  # m/s
            },
            "manipulation": {
                "max_weight": 2.0,  # kg
                "forbidden_objects": ["fragile", "sharp", "hot"],
                "max_force": 50.0  # Newtons
            },
            "communication": {
                "max_volume": 0.8,  # 0-1 scale
                "censored_words": ["inappropriate", "offensive"]
            }
        }

    def check_action_safety(self, action_step: ActionStep, robot_state: Dict,
                           environment_data: Dict) -> Dict[str, Any]:
        """Check if an action is safe to execute"""
        safety_checks = {
            ActionType.NAVIGATE_TO: self._check_navigation_safety,
            ActionType.PICK_UP: self._check_manipulation_safety,
            ActionType.PLACE_OBJECT: self._check_manipulation_safety,
            ActionType.SPEAK: self._check_communication_safety,
        }

        check_func = safety_checks.get(action_step.action)
        if check_func:
            return check_func(action_step, robot_state, environment_data)

        # Default: action is safe
        return {"safe": True, "issues": [], "risk_level": "low"}

    def _check_navigation_safety(self, step: ActionStep, robot_state: Dict,
                                env_data: Dict) -> Dict[str, Any]:
        """Check safety for navigation actions"""
        issues = []

        target_location = step.parameters.get("location")
        if target_location in self.safety_rules["navigation"]["forbidden_areas"]:
            issues.append(f"Navigation to forbidden area: {target_location}")

        # Check path safety (this would interface with navigation stack)
        # For now, we'll assume a simple check
        if env_data.get("obstacles", {}).get(target_location):
            issues.append(f"Path to {target_location} has obstacles")

        # Check distance to humans in path
        humans_nearby = env_data.get("humans", [])
        for human in humans_nearby:
            distance = self._calculate_distance(robot_state["location"], human["position"])
            if distance < self.safety_rules["navigation"]["min_distance_to_human"]:
                issues.append(f"Human detected too close: {distance:.2f}m")

        return {
            "safe": len(issues) == 0,
            "issues": issues,
            "risk_level": "high" if issues else "low"
        }

    def _check_manipulation_safety(self, step: ActionStep, robot_state: Dict,
                                  env_data: Dict) -> Dict[str, Any]:
        """Check safety for manipulation actions"""
        issues = []

        obj_name = step.parameters.get("object", "")

        # Check if object is forbidden
        if obj_name in self.safety_rules["manipulation"]["forbidden_objects"]:
            issues.append(f"Object is forbidden for manipulation: {obj_name}")

        # Check object weight (if available)
        obj_weight = env_data.get("objects", {}).get(obj_name, {}).get("weight")
        if obj_weight and obj_weight > self.safety_rules["manipulation"]["max_weight"]:
            issues.append(f"Object too heavy: {obj_weight}kg > {self.safety_rules['manipulation']['max_weight']}kg")

        return {
            "safe": len(issues) == 0,
            "issues": issues,
            "risk_level": "high" if issues else "low"
        }

    def _check_communication_safety(self, step: ActionStep, robot_state: Dict,
                                   env_data: Dict) -> Dict[str, Any]:
        """Check safety for communication actions"""
        issues = []

        text = step.parameters.get("text", "")

        # Check for censored words
        for word in self.safety_rules["communication"]["censored_words"]:
            if word.lower() in text.lower():
                issues.append(f"Text contains censored word: {word}")

        # Check volume level
        volume = step.parameters.get("volume", 1.0)
        if volume > self.safety_rules["communication"]["max_volume"]:
            issues.append(f"Volume too loud: {volume} > {self.safety_rules['communication']['max_volume']}")

        return {
            "safe": len(issues) == 0,
            "issues": issues,
            "risk_level": "high" if issues else "low"
        }

    def _calculate_distance(self, pos1: Dict, pos2: Dict) -> float:
        """Calculate distance between two positions"""
        # Simple 2D distance calculation
        dx = pos1.get("x", 0) - pos2.get("x", 0)
        dy = pos1.get("y", 0) - pos2.get("y", 0)
        return (dx**2 + dy**2)**0.5
```

### Plan Approval Workflow
Implement a workflow for approving LLM-generated plans:

```python
from enum import Enum

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"

class PlanApprovalSystem:
    def __init__(self, safety_framework: SafetyFramework):
        self.safety_framework = safety_framework
        self.approval_threshold = 0.7  # Minimum confidence for auto-approval
        self.high_risk_actions = [ActionType.NAVIGATE_TO, ActionType.PICK_UP, ActionType.PLACE_OBJECT]

    def approve_plan(self, plan: ActionPlan, robot_state: Dict,
                    environment_data: Dict) -> ApprovalStatus:
        """Determine if a plan should be approved for execution"""

        # Check overall plan confidence
        if plan.confidence < self.approval_threshold:
            return ApprovalStatus.PENDING  # Need human review

        # Check for high-risk actions
        has_high_risk = any(step.action in self.high_risk_actions for step in plan.steps)
        if has_high_risk:
            # Perform detailed safety checks
            for i, step in enumerate(plan.steps):
                safety_result = self.safety_framework.check_action_safety(
                    step, robot_state, environment_data
                )

                if not safety_result["safe"]:
                    if safety_result["risk_level"] == "high":
                        return ApprovalStatus.REJECTED
                    else:
                        return ApprovalStatus.PENDING  # Medium risk, human review needed

        # If we get here, the plan is safe and high confidence
        return ApprovalStatus.APPROVED

    def suggest_plan_modifications(self, plan: ActionPlan, robot_state: Dict,
                                  environment_data: Dict) -> List[Dict[str, Any]]:
        """Suggest modifications to make a plan safer"""
        suggestions = []

        for i, step in enumerate(plan.steps):
            safety_result = self.safety_framework.check_action_safety(
                step, robot_state, environment_data
            )

            if not safety_result["safe"]:
                for issue in safety_result["issues"]:
                    suggestions.append({
                        "step_index": i,
                        "original_action": step.action.value,
                        "issue": issue,
                        "suggestion": self._generate_suggestion_for_issue(issue, step)
                    })

        return suggestions

    def _generate_suggestion_for_issue(self, issue: str, step: ActionStep) -> str:
        """Generate a suggestion to address a specific safety issue"""
        if "forbidden area" in issue.lower():
            return f"Avoid navigating to restricted area. Consider alternative route."
        elif "object too heavy" in issue.lower():
            return f"Object exceeds weight limit. Find lighter alternative or request assistance."
        elif "human detected too close" in issue.lower():
            return f"Wait for human to move away or navigate around safely."
        else:
            return f"Review action parameters for safety compliance."
```

## Performance Optimization

### Caching Strategies
Implement caching to improve response times:

```python
import hashlib
from functools import lru_cache
import time

class PlanCache:
    def __init__(self, max_size: int = 128, ttl: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds

    def get(self, key: str) -> tuple:
        """Get cached plan with TTL check"""
        if key in self.cache:
            cached_time, plan = self.cache[key]
            if time.time() - cached_time < self.ttl:
                return True, plan
            else:
                # Remove expired entry
                del self.cache[key]

        return False, None

    def put(self, key: str, plan) -> None:
        """Put plan in cache"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
            del self.cache[oldest_key]

        self.cache[key] = (time.time(), plan)

    def generate_key(self, command: str, robot_state: Dict) -> str:
        """Generate cache key from command and state"""
        combined = f"{command}_{hash(str(sorted(robot_state.items())))}"
        return hashlib.md5(combined.encode()).hexdigest()

class OptimizedCognitivePlanner:
    def __init__(self, llm_planner, cache_ttl: int = 3600):
        self.llm_planner = llm_planner
        self.cache = PlanCache(ttl=cache_ttl)
        self.validator = PlanValidator()

    async def generate_plan(self, command: str, robot_state: Dict) -> Dict:
        """Generate plan with caching"""
        # Generate cache key
        cache_key = self.cache.generate_key(command, robot_state)

        # Check cache first
        cached, cached_plan = self.cache.get(cache_key)
        if cached:
            print("Retrieved plan from cache")
            return cached_plan

        # Generate new plan
        plan = await self.llm_planner.generate_plan(command, robot_state)

        # Validate plan
        validation = self.validator.validate_plan(plan, robot_state)
        if validation["valid"]:
            # Add confidence adjustment
            adjusted_confidence = max(0.0, min(1.0, plan.confidence + validation["confidence_adjustment"]))
            plan.confidence = adjusted_confidence

            # Cache the plan
            self.cache.put(cache_key, plan)

        return plan
```

### Asynchronous Processing
Use async/await for better performance:

```python
import asyncio
import concurrent.futures
from typing import List

class AsyncCognitiveSystem:
    def __init__(self, llm_planner, num_workers: int = 4):
        self.llm_planner = llm_planner
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        self.cache = PlanCache()

    async def process_multiple_commands(self, commands: List[str], robot_state: Dict) -> List[Dict]:
        """Process multiple commands concurrently"""
        tasks = [
            asyncio.create_task(self._process_single_command(cmd, robot_state))
            for cmd in commands
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out any exceptions
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                print(f"Command processing failed: {result}")
            else:
                valid_results.append(result)

        return valid_results

    async def _process_single_command(self, command: str, robot_state: Dict) -> Dict:
        """Process a single command asynchronously"""
        loop = asyncio.get_event_loop()

        # Check cache first
        cache_key = self.cache.generate_key(command, robot_state)
        cached, cached_plan = self.cache.get(cache_key)
        if cached:
            return cached_plan

        # Generate plan using thread pool for I/O operations
        plan = await loop.run_in_executor(
            self.executor,
            lambda: asyncio.run(self.llm_planner.generate_plan(command, robot_state))
        )

        # Cache the result
        self.cache.put(cache_key, plan)

        return plan
```

## Troubleshooting Common Issues

### Rate Limiting and API Errors
Handle LLM API limitations:

```python
import time
import random
from typing import Optional

class RobustLLMInterface:
    def __init__(self, base_planner, max_retries: int = 3, base_delay: float = 1.0):
        self.base_planner = base_planner
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def generate_plan_with_retry(self, command: str, robot_state: Dict) -> Optional[Dict]:
        """Generate plan with retry logic for API errors"""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return await self.base_planner.generate_plan(command, robot_state)
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()

                if "rate limit" in error_msg or "quota" in error_msg:
                    # Rate limit error - use exponential backoff
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limit hit, waiting {delay:.2f}s before retry {attempt + 1}/{self.max_retries}")
                    await asyncio.sleep(delay)
                elif "timeout" in error_msg:
                    # Timeout - shorter delay
                    delay = self.base_delay * (1.5 ** attempt)
                    print(f"Timeout, waiting {delay:.2f}s before retry {attempt + 1}/{self.max_retries}")
                    await asyncio.sleep(delay)
                else:
                    # Other error - don't retry
                    break

        print(f"All retry attempts failed: {last_exception}")
        return None
```

### Context Length Management
Handle long conversations and context limits:

```python
class ContextManager:
    def __init__(self, max_context_length: int = 4096):
        self.max_context_length = max_context_length
        self.conversation_history = []
        self.summary = ""

    def add_interaction(self, user_input: str, ai_response: str) -> None:
        """Add a user-AI interaction to the context"""
        interaction = {
            "user": user_input,
            "ai": ai_response,
            "timestamp": time.time()
        }

        self.conversation_history.append(interaction)

        # Trim history if it's getting too long
        self._trim_history()

    def get_context_for_prompt(self) -> str:
        """Get the context to include in the next prompt"""
        if len(self.conversation_history) == 0:
            return ""

        # If history is short, include all of it
        if self._estimate_token_count(str(self.conversation_history)) < self.max_context_length // 2:
            return str(self.conversation_history)

        # Otherwise, include a summary plus recent interactions
        recent_count = 5  # Keep last 5 interactions
        recent_history = self.conversation_history[-recent_count:]

        context = f"Previous conversation summary: {self.summary}\n\n"
        context += "Recent interactions:\n"
        for i, interaction in enumerate(recent_history, 1):
            context += f"{i}. User: {interaction['user']}\n"
            context += f"   AI: {interaction['ai']}\n\n"

        return context

    def _estimate_token_count(self, text: str) -> int:
        """Roughly estimate token count (1 token ~ 4 characters)"""
        return len(text) // 4

    def _trim_history(self) -> None:
        """Trim conversation history to stay within limits"""
        while self._estimate_token_count(str(self.conversation_history)) > self.max_context_length:
            if len(self.conversation_history) <= 1:
                break

            # Remove the oldest interaction
            removed = self.conversation_history.pop(0)

            # Update summary with removed content
            self.summary += f" Earlier: User said '{removed['user']}', AI responded '{removed['ai']}' "
```

## Best Practices

### 1. Model Selection Strategy
Choose the right LLM for your specific use case:
- **GPT-4**: Best for complex reasoning and long contexts
- **GPT-3.5**: Good balance of performance and cost
- **Claude**: Strong safety features and reasoning
- **Open-source models**: For privacy or offline applications

### 2. Safety-First Design
Always prioritize safety in your implementation:
- Implement multiple layers of safety checks
- Use conservative confidence thresholds
- Provide human oversight capabilities
- Design graceful failure modes

### 3. Performance Optimization
- Use caching for common commands
- Implement async processing
- Optimize prompt length
- Consider local model deployment for critical operations

### 4. Continuous Learning
- Log interactions for analysis
- Implement feedback mechanisms
- Monitor plan execution success rates
- Continuously refine prompts and validation rules

### 5. Error Handling
- Implement comprehensive error handling
- Provide fallback mechanisms
- Log errors for debugging
- Design for resilience

## Exercises

1. Set up an LLM integration with your preferred API (OpenAI, Anthropic, etc.)
2. Create a prompt engineering framework for robotic applications
3. Implement a plan validation system with safety checks
4. Build a ROS 2 node that integrates LLM planning with robot execution
5. Add caching and performance optimization to your system
6. Implement error handling and retry mechanisms for API calls
7. Create a safety framework with multiple validation layers
8. Design and implement a context management system for maintaining conversation history
9. Build a plan approval workflow with human oversight capabilities
10. Create a fallback mechanism for when LLM calls fail or return low-confidence results

## Next Steps

After completing this chapter, proceed to learn about natural language to ROS 2 action translation to understand how to convert LLM outputs into specific robot commands.