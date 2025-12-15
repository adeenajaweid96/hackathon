# Example Implementations: Voice Commands for Humanoid Robots

## Overview
This chapter provides complete, practical example implementations of voice command processing systems for humanoid robots. These examples demonstrate the integration of Whisper speech recognition, LLM cognitive planning, and ROS 2 action execution in real-world scenarios.

## Learning Objectives
- Implement complete voice command processing pipelines
- Integrate multiple VLA components into cohesive systems
- Create practical applications for humanoid robot interaction
- Understand real-world challenges and solutions
- Deploy voice command systems on actual hardware

## Prerequisites
- Completed Whisper integration chapter
- Completed LLM cognitive planning chapter
- Completed multimodal interaction chapter
- Completed natural language to ROS action chapter
- Basic understanding of ROS 2 concepts

## Table of Contents
1. [Simple Voice Command System](#simple-voice-command-system)
2. [Advanced Voice Command with Context](#advanced-voice-command-with-context)
3. [Multimodal Voice Commands](#multimodal-voice-commands)
4. [Safety-Enhanced Voice Commands](#safety-enhanced-voice-commands)
5. [Real-World Deployment Examples](#real-world-deployment-examples)
6. [Performance Optimization Examples](#performance-optimization-examples)
7. [Troubleshooting Common Issues](#troubleshooting-common-issues)

## Simple Voice Command System

Let's start with a basic voice command system that integrates all core components:

```python
#!/usr/bin/env python3
"""
Simple Voice Command System for Humanoid Robot

This example demonstrates a basic voice command processing pipeline
that integrates Whisper, LLM cognitive planning, and ROS 2 action execution.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import AudioData
from vision_language_action_msgs.msg import SpeechRecognitionResult
import whisper
import openai
import json
import time
import threading
import queue
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class CommandResult:
    success: bool
    message: str
    action_taken: str


class SimpleVoiceCommandNode(Node):
    def __init__(self):
        super().__init__('simple_voice_command_node')

        # Initialize Whisper model
        self.whisper_model = whisper.load_model("small")

        # Initialize command queue
        self.command_queue = queue.Queue()

        # Create publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.speech_pub = self.create_publisher(String, '/robot_speech', 10)

        # Create subscribers
        self.audio_sub = self.create_subscription(
            AudioData,
            'audio_input',
            self.audio_callback,
            10
        )

        self.get_logger().info("Simple Voice Command Node initialized")

        # Start command processing thread
        self.command_thread = threading.Thread(target=self.process_commands)
        self.command_thread.daemon = True
        self.command_thread.start()

    def audio_callback(self, msg):
        """Process incoming audio and generate text"""
        try:
            # Convert audio data to numpy array
            audio_array = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe audio using Whisper
            result = self.whisper_model.transcribe(audio_array, language="en")
            recognized_text = result["text"].strip()

            if recognized_text:
                self.get_logger().info(f"Recognized: {recognized_text}")

                # Add to command queue for processing
                self.command_queue.put(recognized_text)

        except Exception as e:
            self.get_logger().error(f"Error processing audio: {e}")

    def process_commands(self):
        """Process commands from the queue"""
        while rclpy.ok():
            try:
                # Get command from queue with timeout
                command = self.command_queue.get(timeout=1.0)

                # Process the command
                self.execute_command(command)

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Error processing command: {e}")

    def execute_command(self, text: str):
        """Execute a recognized voice command"""
        try:
            # Simple command mapping
            if "forward" in text.lower():
                self.move_forward()
                self.speak("Moving forward")
            elif "backward" in text.lower():
                self.move_backward()
                self.speak("Moving backward")
            elif "left" in text.lower():
                self.turn_left()
                self.speak("Turning left")
            elif "right" in text.lower():
                self.turn_right()
                self.speak("Turning right")
            elif "stop" in text.lower():
                self.stop_robot()
                self.speak("Stopping")
            elif "hello" in text.lower() or "hi" in text.lower():
                self.speak("Hello! How can I help you?")
            else:
                # Use LLM for more complex commands
                result = self.process_complex_command(text)
                if result:
                    self.speak(result)
                else:
                    self.speak("I didn't understand that command")

        except Exception as e:
            self.get_logger().error(f"Error executing command: {e}")
            self.speak("Sorry, I encountered an error processing that command")

    def move_forward(self):
        """Move robot forward"""
        msg = Twist()
        msg.linear.x = 0.2  # m/s
        self.cmd_vel_pub.publish(msg)

    def move_backward(self):
        """Move robot backward"""
        msg = Twist()
        msg.linear.x = -0.2  # m/s
        self.cmd_vel_pub.publish(msg)

    def turn_left(self):
        """Turn robot left"""
        msg = Twist()
        msg.angular.z = 0.5  # rad/s
        self.cmd_vel_pub.publish(msg)

    def turn_right(self):
        """Turn robot right"""
        msg = Twist()
        msg.angular.z = -0.5  # rad/s
        self.cmd_vel_pub.publish(msg)

    def stop_robot(self):
        """Stop robot movement"""
        msg = Twist()
        self.cmd_vel_pub.publish(msg)

    def speak(self, text: str):
        """Publish text for robot to speak"""
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)

    def process_complex_command(self, text: str) -> Optional[str]:
        """Process complex commands using LLM"""
        try:
            # Simple LLM-based command processing
            # In a real implementation, you would use a more sophisticated approach
            if any(word in text.lower() for word in ["dance", "dancing", "move"]):
                self.perform_dance()
                return "Dancing for you!"
            elif any(word in text.lower() for word in ["wave", "hello", "greet"]):
                self.wave_hello()
                return "Waving hello!"
            else:
                return None
        except Exception as e:
            self.get_logger().error(f"Error processing complex command: {e}")
            return None

    def perform_dance(self):
        """Perform a simple dance movement"""
        # This would control the robot's joints for dancing
        self.get_logger().info("Performing dance movement")

    def wave_hello(self):
        """Wave hello gesture"""
        # This would control the robot's arm for waving
        self.get_logger().info("Waving hello")


def main(args=None):
    rclpy.init(args=args)
    node = SimpleVoiceCommandNode()

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

## Advanced Voice Command with Context

Here's a more sophisticated example that maintains context and can handle complex multi-turn conversations:

```python
#!/usr/bin/env python3
"""
Advanced Voice Command System with Context Awareness

This example demonstrates a voice command system that maintains context
and can handle complex multi-turn conversations with the user.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Int8
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import AudioData
from vision_language_action_msgs.msg import SpeechRecognitionResult
import whisper
import openai
import json
import time
import threading
import queue
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import datetime


@dataclass
class ConversationContext:
    last_command: str = ""
    last_response: str = ""
    conversation_history: List[str] = None
    timestamp: datetime.datetime = None
    user_name: str = "user"

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []


class AdvancedVoiceCommandNode(Node):
    def __init__(self):
        super().__init__('advanced_voice_command_node')

        # Initialize Whisper model
        self.whisper_model = whisper.load_model("small")

        # Initialize context
        self.context = ConversationContext()

        # Command queue
        self.command_queue = queue.Queue()

        # Create publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.speech_pub = self.create_publisher(String, '/robot_speech', 10)
        self.context_pub = self.create_publisher(String, '/conversation_context', 10)

        # Create subscribers
        self.audio_sub = self.create_subscription(
            AudioData,
            'audio_input',
            self.audio_callback,
            10
        )

        # Timer for context updates
        self.context_timer = self.create_timer(5.0, self.update_context)

        self.get_logger().info("Advanced Voice Command Node initialized")

        # Start command processing thread
        self.command_thread = threading.Thread(target=self.process_commands)
        self.command_thread.daemon = True
        self.command_thread.start()

    def audio_callback(self, msg):
        """Process incoming audio and generate text"""
        try:
            # Convert audio data to numpy array
            audio_array = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe audio using Whisper
            result = self.whisper_model.transcribe(audio_array, language="en")
            recognized_text = result["text"].strip()

            if recognized_text:
                self.get_logger().info(f"Recognized: {recognized_text}")

                # Update context
                self.context.last_command = recognized_text
                self.context.timestamp = datetime.datetime.now()
                self.context.conversation_history.append(f"User: {recognized_text}")

                # Add to command queue for processing
                self.command_queue.put(recognized_text)

        except Exception as e:
            self.get_logger().error(f"Error processing audio: {e}")

    def process_commands(self):
        """Process commands from the queue"""
        while rclpy.ok():
            try:
                # Get command from queue with timeout
                command = self.command_queue.get(timeout=1.0)

                # Process the command with context
                response = self.execute_command_with_context(command)

                if response:
                    self.speak(response)
                    self.context.last_response = response
                    self.context.conversation_history.append(f"Robot: {response}")

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Error processing command: {e}")

    def execute_command_with_context(self, text: str) -> Optional[str]:
        """Execute a recognized voice command with context awareness"""
        try:
            # Handle context-dependent commands
            if "repeat" in text.lower() or "again" in text.lower():
                return self.repeat_last_action()
            elif "what did I just say" in text.lower() or "what did I say" in text.lower():
                return f"You said: {self.context.last_command}"
            elif "what did you say" in text.lower():
                return f"I said: {self.context.last_response}"
            elif "my name" in text.lower():
                return f"Your name is {self.context.user_name}. How can I help you, {self.context.user_name}?"
            elif "call me" in text.lower():
                return self.set_user_name(text)
            elif "remember" in text.lower():
                return self.remember_information(text)
            elif "recall" in text.lower():
                return self.recall_information(text)
            else:
                # Handle regular commands with context
                return self.execute_regular_command(text)

        except Exception as e:
            self.get_logger().error(f"Error executing command with context: {e}")
            return "Sorry, I encountered an error processing that command"

    def repeat_last_action(self) -> str:
        """Repeat the last action"""
        if self.context.last_command:
            # Re-execute the last command
            response = self.execute_regular_command(self.context.last_command)
            return f"Repeating: {self.context.last_command}. {response}"
        else:
            return "I don't have a previous command to repeat."

    def set_user_name(self, text: str) -> str:
        """Set the user's name from the command"""
        # Extract name from command like "call me John"
        words = text.lower().split()
        try:
            call_index = words.index("call") + 1
            if words[call_index] == "me":
                call_index += 1
            if call_index < len(words):
                name = words[call_index]
                self.context.user_name = name
                return f"Okay, I'll call you {name} from now on."
        except (ValueError, IndexError):
            pass

        return f"I couldn't understand the name in your command."

    def remember_information(self, text: str) -> str:
        """Remember information provided by the user"""
        # Extract information after "remember"
        remember_index = text.lower().find("remember")
        if remember_index != -1:
            info = text[remember_index + len("remember"):].strip()
            if info:
                # Store in context (in a real system, you'd use a database)
                self.get_logger().info(f"Remembering: {info}")
                return f"I'll remember that {info}."

        return "I didn't understand what you want me to remember."

    def recall_information(self, text: str) -> str:
        """Recall information previously remembered"""
        # This is a simple implementation - in a real system, you'd query a knowledge base
        if "my" in text.lower():
            return f"Your name is {self.context.user_name}."
        else:
            return "I don't have specific information to recall."

    def execute_regular_command(self, text: str) -> str:
        """Execute a regular command without special context handling"""
        # Simple command mapping with some context awareness
        if "forward" in text.lower():
            self.move_forward()
            return "Moving forward as requested."
        elif "backward" in text.lower():
            self.move_backward()
            return "Moving backward as requested."
        elif "left" in text.lower():
            self.turn_left()
            return "Turning left as requested."
        elif "right" in text.lower():
            self.turn_right()
            return "Turning right as requested."
        elif "stop" in text.lower():
            self.stop_robot()
            return "Stopping as requested."
        elif "hello" in text.lower() or "hi" in text.lower():
            return f"Hello {self.context.user_name}! How can I help you today?"
        elif "thank" in text.lower():
            return f"You're welcome, {self.context.user_name}!"
        elif "bye" in text.lower() or "goodbye" in text.lower():
            return f"Goodbye {self.context.user_name}! Have a great day!"
        else:
            # Use LLM for more complex commands with context
            result = self.process_complex_command_with_context(text)
            if result:
                return result
            else:
                return f"I'm not sure how to handle '{text}'. Can you rephrase that?"

    def move_forward(self):
        """Move robot forward"""
        msg = Twist()
        msg.linear.x = 0.2  # m/s
        self.cmd_vel_pub.publish(msg)

    def move_backward(self):
        """Move robot backward"""
        msg = Twist()
        msg.linear.x = -0.2  # m/s
        self.cmd_vel_pub.publish(msg)

    def turn_left(self):
        """Turn robot left"""
        msg = Twist()
        msg.angular.z = 0.5  # rad/s
        self.cmd_vel_pub.publish(msg)

    def turn_right(self):
        """Turn robot right"""
        msg = Twist()
        msg.angular.z = -0.5  # rad/s
        self.cmd_vel_pub.publish(msg)

    def stop_robot(self):
        """Stop robot movement"""
        msg = Twist()
        self.cmd_vel_pub.publish(msg)

    def speak(self, text: str):
        """Publish text for robot to speak"""
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)

    def process_complex_command_with_context(self, text: str) -> Optional[str]:
        """Process complex commands using LLM with context"""
        try:
            # Simple context-aware command processing
            # In a real implementation, you would use a more sophisticated approach
            if any(word in text.lower() for word in ["dance", "dancing", "move around"]):
                self.perform_dance()
                return f"Dancing for you, {self.context.user_name}!"
            elif any(word in text.lower() for word in ["wave", "hello", "greet"]):
                self.wave_hello()
                return f"Waving hello, {self.context.user_name}!"
            elif "time" in text.lower():
                current_time = datetime.datetime.now().strftime("%H:%M")
                return f"The current time is {current_time}."
            elif "date" in text.lower():
                current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                return f"Today's date is {current_date}."
            else:
                return None
        except Exception as e:
            self.get_logger().error(f"Error processing complex command with context: {e}")
            return None

    def perform_dance(self):
        """Perform a simple dance movement"""
        # This would control the robot's joints for dancing
        self.get_logger().info("Performing dance movement")

    def wave_hello(self):
        """Wave hello gesture"""
        # This would control the robot's arm for waving
        self.get_logger().info("Waving hello")

    def update_context(self):
        """Periodically update context information"""
        # This could be used to periodically publish context updates
        # or clean up old context information
        context_msg = String()
        context_msg.data = json.dumps({
            "user_name": self.context.user_name,
            "last_command": self.context.last_command,
            "last_response": self.context.last_response,
            "timestamp": self.context.timestamp.isoformat() if self.context.timestamp else None,
            "history_length": len(self.context.conversation_history)
        })
        self.context_pub.publish(context_msg)


def main(args=None):
    rclpy.init(args=args)
    node = AdvancedVoiceCommandNode()

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

## Multimodal Voice Commands

This example demonstrates integration with visual information to create multimodal voice commands:

```python
#!/usr/bin/env python3
"""
Multimodal Voice Command System

This example demonstrates voice commands that incorporate visual information
from cameras to create more sophisticated interaction capabilities.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, AudioData
from geometry_msgs.msg import Twist, Point
from vision_language_action_msgs.msg import SpeechRecognitionResult
import whisper
import numpy as np
import cv2
from cv_bridge import CvBridge
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json


@dataclass
class ObjectDetection:
    name: str
    confidence: float
    position: Point
    bounding_box: tuple  # (x, y, width, height)


class MultimodalVoiceCommandNode(Node):
    def __init__(self):
        super().__init__('multimodal_voice_command_node')

        # Initialize Whisper model
        self.whisper_model = whisper.load_model("small")

        # Initialize OpenCV bridge
        self.bridge = CvBridge()

        # Initialize object detection results
        self.current_objects = []
        self.current_image = None

        # Command queue
        self.command_queue = queue.Queue()

        # Create publishers
        self.cmd_vel_pub = self.create_publisher(Twilt, '/cmd_vel', 10)
        self.speech_pub = self.create_publisher(String, '/robot_speech', 10)
        self.target_pub = self.create_publisher(Point, '/target_location', 10)

        # Create subscribers
        self.audio_sub = self.create_subscription(
            AudioData,
            'audio_input',
            self.audio_callback,
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.get_logger().info("Multimodal Voice Command Node initialized")

        # Start command processing thread
        self.command_thread = threading.Thread(target=self.process_commands)
        self.command_thread.daemon = True
        self.command_thread.start()

    def audio_callback(self, msg):
        """Process incoming audio and generate text"""
        try:
            # Convert audio data to numpy array
            audio_array = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe audio using Whisper
            result = self.whisper_model.transcribe(audio_array, language="en")
            recognized_text = result["text"].strip()

            if recognized_text:
                self.get_logger().info(f"Recognized: {recognized_text}")

                # Add to command queue for processing
                self.command_queue.put(recognized_text)

        except Exception as e:
            self.get_logger().error(f"Error processing audio: {e}")

    def image_callback(self, msg):
        """Process incoming image for object detection"""
        try:
            # Convert ROS image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Store current image
            self.current_image = cv_image

            # Perform simple object detection (in a real system, you'd use a trained model)
            objects = self.detect_objects(cv_image)
            self.current_objects = objects

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def detect_objects(self, image):
        """Simple object detection - in a real system, use a trained model like YOLO"""
        # This is a placeholder implementation
        # In a real system, you'd use a trained object detection model
        objects = []

        # For demonstration, we'll create some mock objects
        # In a real system, you'd run inference with a model
        if image is not None:
            height, width = image.shape[:2]

            # Mock detection of a "ball" in the center
            if width > 100 and height > 100:
                center_obj = ObjectDetection(
                    name="ball",
                    confidence=0.85,
                    position=Point(x=width/2, y=height/2, z=0.0),
                    bounding_box=(width//2 - 50, height//2 - 50, 100, 100)
                )
                objects.append(center_obj)

        return objects

    def process_commands(self):
        """Process commands from the queue"""
        while rclpy.ok():
            try:
                # Get command from queue with timeout
                command = self.command_queue.get(timeout=1.0)

                # Process the command with multimodal context
                response = self.execute_multimodal_command(command)

                if response:
                    self.speak(response)

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Error processing command: {e}")

    def execute_multimodal_command(self, text: str) -> Optional[str]:
        """Execute a voice command with multimodal context"""
        try:
            # Check for commands that reference visual elements
            if "go to" in text.lower() or "move to" in text.lower():
                return self.go_to_object(text)
            elif "find" in text.lower() or "look for" in text.lower():
                return self.find_object(text)
            elif "what is" in text.lower() or "describe" in text.lower():
                return self.describe_scene(text)
            elif "pick up" in text.lower() or "grasp" in text.lower():
                return self.pick_up_object(text)
            else:
                # Handle regular commands
                return self.execute_regular_command(text)

        except Exception as e:
            self.get_logger().error(f"Error executing multimodal command: {e}")
            return "Sorry, I encountered an error processing that command"

    def go_to_object(self, text: str) -> str:
        """Navigate to a specific object"""
        # Extract object name from command
        object_name = self.extract_object_name(text)

        if not object_name:
            return "What object would you like me to go to?"

        # Find the object in current detections
        target_object = self.find_object_by_name(object_name)

        if target_object:
            # Navigate to the object
            self.navigate_to_object(target_object)
            return f"Going to the {object_name}."
        else:
            return f"I don't see a {object_name} right now."

    def find_object(self, text: str) -> str:
        """Find a specific object in the current scene"""
        # Extract object name from command
        object_name = self.extract_object_name(text)

        if not object_name:
            return "What object would you like me to find?"

        # Find the object in current detections
        target_object = self.find_object_by_name(object_name)

        if target_object:
            confidence_percent = int(target_object.confidence * 100)
            return f"I found a {object_name} with {confidence_percent}% confidence."
        else:
            return f"I don't see a {object_name} right now."

    def describe_scene(self, text: str) -> str:
        """Describe the current scene"""
        if not self.current_objects:
            return "I don't see any objects right now."

        object_names = [obj.name for obj in self.current_objects]
        unique_names = list(set(object_names))

        if len(unique_names) == 1:
            return f"I see a {unique_names[0]}."
        else:
            objects_str = ", ".join(unique_names[:-1]) + f" and a {unique_names[-1]}"
            return f"I see {objects_str}."

    def pick_up_object(self, text: str) -> str:
        """Pick up a specific object"""
        # Extract object name from command
        object_name = self.extract_object_name(text)

        if not object_name:
            return "What object would you like me to pick up?"

        # Find the object in current detections
        target_object = self.find_object_by_name(object_name)

        if target_object:
            # In a real system, this would trigger manipulation
            return f"Picking up the {object_name}."
        else:
            return f"I don't see a {object_name} to pick up right now."

    def extract_object_name(self, text: str) -> Optional[str]:
        """Extract object name from command text"""
        # Simple extraction - in a real system, use NLP
        text_lower = text.lower()

        # Common object names to look for
        common_objects = ["ball", "cup", "box", "book", "bottle", "person", "chair", "table"]

        for obj in common_objects:
            if obj in text_lower:
                return obj

        return None

    def find_object_by_name(self, name: str) -> Optional[ObjectDetection]:
        """Find an object by name in current detections"""
        for obj in self.current_objects:
            if obj.name.lower() == name.lower():
                return obj
        return None

    def navigate_to_object(self, obj: ObjectDetection):
        """Navigate to a specific object"""
        # Calculate movement based on object position
        if self.current_image is not None:
            height, width = self.current_image.shape[:2]

            # Calculate how to move based on object position
            center_x = width / 2
            obj_x = obj.position.x

            msg = Twist()

            # Move forward if object is visible
            msg.linear.x = 0.1

            # Turn toward object if not centered
            if obj_x < center_x - width * 0.1:  # Left of center
                msg.angular.z = 0.2
            elif obj_x > center_x + width * 0.1:  # Right of center
                msg.angular.z = -0.2
            else:  # Centered
                msg.angular.z = 0.0

            self.cmd_vel_pub.publish(msg)

    def execute_regular_command(self, text: str) -> str:
        """Execute a regular command without special multimodal handling"""
        # Simple command mapping
        if "forward" in text.lower():
            self.move_forward()
            return "Moving forward."
        elif "backward" in text.lower():
            self.move_backward()
            return "Moving backward."
        elif "left" in text.lower():
            self.turn_left()
            return "Turning left."
        elif "right" in text.lower():
            self.turn_right()
            return "Turning right."
        elif "stop" in text.lower():
            self.stop_robot()
            return "Stopping."
        elif "hello" in text.lower() or "hi" in text.lower():
            return "Hello! How can I help you?"
        else:
            return f"I'm not sure how to handle '{text}'. Can you rephrase that?"

    def move_forward(self):
        """Move robot forward"""
        msg = Twist()
        msg.linear.x = 0.2  # m/s
        self.cmd_vel_pub.publish(msg)

    def move_backward(self):
        """Move robot backward"""
        msg = Twist()
        msg.linear.x = -0.2  # m/s
        self.cmd_vel_pub.publish(msg)

    def turn_left(self):
        """Turn robot left"""
        msg = Twist()
        msg.angular.z = 0.5  # rad/s
        self.cmd_vel_pub.publish(msg)

    def turn_right(self):
        """Turn robot right"""
        msg = Twist()
        msg.angular.z = -0.5  # rad/s
        self.cmd_vel_pub.publish(msg)

    def stop_robot(self):
        """Stop robot movement"""
        msg = Twist()
        self.cmd_vel_pub.publish(msg)

    def speak(self, text: str):
        """Publish text for robot to speak"""
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MultimodalVoiceCommandNode()

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

## Safety-Enhanced Voice Commands

This example demonstrates how to implement safety features in voice command systems:

```python
#!/usr/bin/env python3
"""
Safety-Enhanced Voice Command System

This example demonstrates voice command processing with comprehensive
safety checks and validation to ensure safe robot operation.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import LaserScan, Image, AudioData
from geometry_msgs.msg import Twist, Point
from vision_language_action_msgs.msg import SpeechRecognitionResult
import whisper
import numpy as np
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import json
import datetime


@dataclass
class SafetyState:
    emergency_stop: bool = False
    obstacle_detected: bool = False
    obstacle_distance: float = 0.0
    safe_zone: bool = True
    last_action_time: datetime.datetime = None
    safety_violation_count: int = 0


class SafetyEnhancedVoiceCommandNode(Node):
    def __init__(self):
        super().__init__('safety_enhanced_voice_command_node')

        # Initialize Whisper model
        self.whisper_model = whisper.load_model("small")

        # Initialize safety state
        self.safety_state = SafetyState()

        # Command queue
        self.command_queue = queue.Queue()
        self.safety_queue = queue.Queue()

        # Create publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.speech_pub = self.create_publisher(String, '/robot_speech', 10)
        self.safety_status_pub = self.create_publisher(String, '/safety_status', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, '/emergency_stop', 10)

        # Create subscribers
        self.audio_sub = self.create_subscription(
            AudioData,
            'audio_input',
            self.audio_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Timer for safety monitoring
        self.safety_timer = self.create_timer(0.1, self.safety_monitor)

        self.get_logger().info("Safety-Enhanced Voice Command Node initialized")

        # Start command processing thread
        self.command_thread = threading.Thread(target=self.process_commands)
        self.command_thread.daemon = True
        self.command_thread.start()

    def audio_callback(self, msg):
        """Process incoming audio and generate text"""
        try:
            # Check if emergency stop is active
            if self.safety_state.emergency_stop:
                self.speak("Emergency stop is active. Please reset before continuing.")
                return

            # Convert audio data to numpy array
            audio_array = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe audio using Whisper
            result = self.whisper_model.transcribe(audio_array, language="en")
            recognized_text = result["text"].strip()

            if recognized_text:
                self.get_logger().info(f"Recognized: {recognized_text}")

                # Add to command queue for processing
                self.command_queue.put(recognized_text)

        except Exception as e:
            self.get_logger().error(f"Error processing audio: {e}")

    def scan_callback(self, msg):
        """Process laser scan for obstacle detection"""
        try:
            # Find minimum distance in scan
            if len(msg.ranges) > 0:
                valid_ranges = [r for r in msg.ranges if r > msg.range_min and r < msg.range_max]
                if valid_ranges:
                    min_distance = min(valid_ranges)
                    self.safety_state.obstacle_distance = min_distance
                    self.safety_state.obstacle_detected = min_distance < 0.5  # 50cm threshold
                else:
                    self.safety_state.obstacle_detected = False
                    self.safety_state.obstacle_distance = float('inf')
            else:
                self.safety_state.obstacle_detected = False
                self.safety_state.obstacle_distance = float('inf')

        except Exception as e:
            self.get_logger().error(f"Error processing scan: {e}")

    def safety_monitor(self):
        """Monitor safety state and publish updates"""
        try:
            # Check for safety violations
            if self.safety_state.obstacle_detected:
                # Stop robot if obstacle is too close
                stop_msg = Twist()
                self.cmd_vel_pub.publish(stop_msg)

                # Publish safety status
                safety_msg = String()
                safety_msg.data = json.dumps({
                    "status": "obstacle_detected",
                    "distance": self.safety_state.obstacle_distance,
                    "timestamp": datetime.datetime.now().isoformat()
                })
                self.safety_status_pub.publish(safety_msg)

            else:
                # Publish safety status
                safety_msg = String()
                safety_msg.data = json.dumps({
                    "status": "safe",
                    "distance": self.safety_state.obstacle_distance,
                    "timestamp": datetime.datetime.now().isoformat()
                })
                self.safety_status_pub.publish(safety_msg)

        except Exception as e:
            self.get_logger().error(f"Error in safety monitor: {e}")

    def process_commands(self):
        """Process commands from the queue"""
        while rclpy.ok():
            try:
                # Get command from queue with timeout
                command = self.command_queue.get(timeout=1.0)

                # Validate safety before executing command
                if self.is_safe_to_execute(command):
                    response = self.execute_safe_command(command)
                    if response:
                        self.speak(response)
                else:
                    self.speak("Command rejected for safety reasons.")

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Error processing command: {e}")

    def is_safe_to_execute(self, command: str) -> bool:
        """Check if it's safe to execute a command"""
        try:
            # Check emergency stop
            if self.safety_state.emergency_stop:
                return False

            # Check for movement commands when obstacles are detected
            movement_keywords = ["forward", "backward", "move", "go", "drive"]
            if any(keyword in command.lower() for keyword in movement_keywords):
                if self.safety_state.obstacle_detected and self.safety_state.obstacle_distance < 0.5:
                    self.get_logger().warn(f"Movement command blocked due to obstacle at {self.safety_state.obstacle_distance}m")
                    return False

            # Check for unsafe commands
            unsafe_keywords = ["harm", "dangerous", "unsafe"]
            if any(keyword in command.lower() for keyword in unsafe_keywords):
                self.get_logger().warn(f"Unsafe command detected: {command}")
                return False

            return True

        except Exception as e:
            self.get_logger().error(f"Error checking safety: {e}")
            return False

    def execute_safe_command(self, text: str) -> Optional[str]:
        """Execute a command with safety validation"""
        try:
            # Update last action time
            self.safety_state.last_action_time = datetime.datetime.now()

            # Simple command mapping with safety checks
            if "forward" in text.lower():
                if self.safety_state.obstacle_detected and self.safety_state.obstacle_distance < 0.5:
                    return "Cannot move forward, obstacle detected."
                else:
                    self.move_forward()
                    return "Moving forward safely."
            elif "backward" in text.lower():
                self.move_backward()
                return "Moving backward safely."
            elif "left" in text.lower():
                self.turn_left()
                return "Turning left safely."
            elif "right" in text.lower():
                self.turn_right()
                return "Turning right safely."
            elif "stop" in text.lower():
                self.stop_robot()
                return "Stopping safely."
            elif "emergency stop" in text.lower() or "panic" in text.lower():
                self.activate_emergency_stop()
                return "Emergency stop activated. Robot stopped."
            elif "reset" in text.lower() or "resume" in text.lower():
                self.reset_emergency_stop()
                return "Emergency stop reset. Ready to continue."
            elif "hello" in text.lower() or "hi" in text.lower():
                return "Hello! I'm ready to help safely."
            elif "status" in text.lower() or "safe" in text.lower():
                return self.get_safety_status()
            else:
                # Use LLM for more complex commands with safety validation
                result = self.process_complex_command_safely(text)
                if result:
                    return result
                else:
                    return f"I'm not sure how to handle '{text}'. Can you rephrase that safely?"

        except Exception as e:
            self.safety_state.safety_violation_count += 1
            self.get_logger().error(f"Error executing safe command: {e}")
            return "Sorry, I encountered an error processing that command safely"

    def activate_emergency_stop(self):
        """Activate emergency stop"""
        self.safety_state.emergency_stop = True

        # Stop robot immediately
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)

        # Publish emergency stop
        emergency_msg = Bool()
        emergency_msg.data = True
        self.emergency_stop_pub.publish(emergency_msg)

    def reset_emergency_stop(self):
        """Reset emergency stop"""
        self.safety_state.emergency_stop = False

        # Publish emergency stop reset
        emergency_msg = Bool()
        emergency_msg.data = False
        self.emergency_stop_pub.publish(emergency_msg)

    def get_safety_status(self) -> str:
        """Get current safety status"""
        status = "System is currently safe. "
        if self.safety_state.emergency_stop:
            status = "EMERGENCY STOP ACTIVE. "
        elif self.safety_state.obstacle_detected:
            status += f"Obstacle detected at {self.safety_state.obstacle_distance:.2f} meters. "

        status += f"{self.safety_state.safety_violation_count} safety violations recorded."
        return status

    def move_forward(self):
        """Move robot forward"""
        if not self.safety_state.obstacle_detected or self.safety_state.obstacle_distance > 0.5:
            msg = Twist()
            msg.linear.x = 0.2  # m/s
            self.cmd_vel_pub.publish(msg)

    def move_backward(self):
        """Move robot backward"""
        msg = Twist()
        msg.linear.x = -0.2  # m/s
        self.cmd_vel_pub.publish(msg)

    def turn_left(self):
        """Turn robot left"""
        msg = Twist()
        msg.angular.z = 0.5  # rad/s
        self.cmd_vel_pub.publish(msg)

    def turn_right(self):
        """Turn robot right"""
        msg = Twist()
        msg.angular.z = -0.5  # rad/s
        self.cmd_vel_pub.publish(msg)

    def stop_robot(self):
        """Stop robot movement"""
        msg = Twist()
        self.cmd_vel_pub.publish(msg)

    def speak(self, text: str):
        """Publish text for robot to speak"""
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)

    def process_complex_command_safely(self, text: str) -> Optional[str]:
        """Process complex commands with safety validation"""
        try:
            # Simple safety-aware command processing
            # In a real implementation, you would use a more sophisticated approach
            if any(word in text.lower() for word in ["dance", "dancing", "move"]):
                if self.safety_state.obstacle_detected and self.safety_state.obstacle_distance < 0.8:
                    return "Cannot dance, area not clear."
                else:
                    self.perform_dance()
                    return "Dancing safely!"
            else:
                return None
        except Exception as e:
            self.get_logger().error(f"Error processing complex command safely: {e}")
            return None

    def perform_dance(self):
        """Perform a simple dance movement"""
        # This would control the robot's joints for dancing
        # Safety checks would be performed before executing
        self.get_logger().info("Performing safe dance movement")


def main(args=None):
    rclpy.init(args=args)
    node = SafetyEnhancedVoiceCommandNode()

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

## Real-World Deployment Examples

This section provides examples of deploying voice command systems in real-world scenarios:

```python
#!/usr/bin/env python3
"""
Real-World Deployment Examples

This example demonstrates how to deploy voice command systems
in real-world environments with practical considerations.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import AudioData, LaserScan
from geometry_msgs.msg import Twist
from vision_language_action_msgs.msg import SpeechRecognitionResult
import whisper
import numpy as np
import threading
import queue
import time
import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import json
import datetime


@dataclass
class DeploymentConfig:
    """Configuration for real-world deployment"""
    model_size: str = "small"
    device: str = "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"
    audio_sample_rate: int = 16000
    confidence_threshold: float = 0.7
    command_timeout: float = 30.0
    max_retries: int = 3
    log_level: str = "INFO"
    enable_logging: bool = True


class RealWorldDeploymentNode(Node):
    def __init__(self):
        super().__init__('real_world_deployment_node')

        # Load deployment configuration
        self.config = DeploymentConfig()

        # Setup logging
        if self.config.enable_logging:
            logging.basicConfig(
                level=getattr(logging, self.config.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('voice_command_system.log'),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None

        # Initialize Whisper model based on configuration
        self.logger.info(f"Loading Whisper model: {self.config.model_size} on {self.config.device}")
        self.whisper_model = whisper.load_model(self.config.model_size).to(self.config.device)

        # Initialize state
        self.command_history = []
        self.performance_metrics = {
            "total_commands": 0,
            "successful_commands": 0,
            "avg_recognition_time": 0.0,
            "avg_confidence": 0.0
        }

        # Command queue
        self.command_queue = queue.Queue()

        # Create publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.speech_pub = self.create_publisher(String, '/robot_speech', 10)
        self.metrics_pub = self.create_publisher(String, '/performance_metrics', 10)

        # Create subscribers
        self.audio_sub = self.create_subscription(
            AudioData,
            'audio_input',
            self.audio_callback,
            10
        )

        # Timer for metrics publishing
        self.metrics_timer = self.create_timer(10.0, self.publish_metrics)

        self.get_logger().info("Real-World Deployment Node initialized")

        # Start command processing thread
        self.command_thread = threading.Thread(target=self.process_commands)
        self.command_thread.daemon = True
        self.command_thread.start()

    def audio_callback(self, msg):
        """Process incoming audio with real-world considerations"""
        try:
            start_time = time.time()

            # Convert audio data to numpy array
            audio_array = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe audio using Whisper
            result = self.whisper_model.transcribe(audio_array, language="en")
            recognized_text = result["text"].strip()

            # Calculate confidence
            confidence = self.calculate_confidence(result)

            if recognized_text and confidence >= self.config.confidence_threshold:
                self.get_logger().info(f"Recognized (confidence {confidence:.2f}): {recognized_text}")

                # Log the recognition
                if self.logger:
                    self.logger.info(f"Recognized: '{recognized_text}' (confidence: {confidence:.2f})")

                # Add to command history
                self.command_history.append({
                    "text": recognized_text,
                    "confidence": confidence,
                    "timestamp": datetime.datetime.now(),
                    "processing_time": time.time() - start_time
                })

                # Update metrics
                self.performance_metrics["total_commands"] += 1
                self.performance_metrics["avg_confidence"] = (
                    (self.performance_metrics["avg_confidence"] * (self.performance_metrics["total_commands"] - 1) + confidence) /
                    self.performance_metrics["total_commands"]
                )

                # Add to command queue for processing
                self.command_queue.put(recognized_text)

            elif recognized_text:
                self.get_logger().warn(f"Low confidence recognition (confidence {confidence:.2f}): {recognized_text}")

                # Log low confidence recognition
                if self.logger:
                    self.logger.warning(f"Low confidence: '{recognized_text}' (confidence: {confidence:.2f})")

        except Exception as e:
            self.get_logger().error(f"Error processing audio: {e}")
            if self.logger:
                self.logger.error(f"Audio processing error: {e}")

    def calculate_confidence(self, result) -> float:
        """Calculate confidence score for recognition result"""
        # This is a simplified confidence calculation
        # In a real system, you'd use more sophisticated methods
        compression_ratio = result.get("compression_ratio", 1.0)

        if compression_ratio > 2.5:
            return 0.3  # Low confidence
        elif compression_ratio > 1.8:
            return 0.7  # Medium confidence
        else:
            return 0.9  # High confidence

    def process_commands(self):
        """Process commands with real-world error handling"""
        while rclpy.ok():
            try:
                # Get command from queue with timeout
                command = self.command_queue.get(timeout=1.0)

                # Process the command with error handling
                start_time = time.time()
                response = self.execute_command_with_retry(command)
                processing_time = time.time() - start_time

                if response:
                    self.speak(response)

                    # Update success metrics
                    self.performance_metrics["successful_commands"] += 1
                    old_avg_time = self.performance_metrics["avg_recognition_time"]
                    total_processed = self.performance_metrics["successful_commands"]
                    self.performance_metrics["avg_recognition_time"] = (
                        (old_avg_time * (total_processed - 1) + processing_time) / total_processed
                    )

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Error processing command: {e}")
                if self.logger:
                    self.logger.error(f"Command processing error: {e}")

    def execute_command_with_retry(self, text: str, max_retries: int = None) -> Optional[str]:
        """Execute command with retry logic"""
        if max_retries is None:
            max_retries = self.config.max_retries

        for attempt in range(max_retries):
            try:
                response = self.execute_command(text)
                return response
            except Exception as e:
                self.get_logger().warn(f"Command execution attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:  # Last attempt
                    if self.logger:
                        self.logger.error(f"Command failed after {max_retries} attempts: {e}")
                    return "Sorry, I couldn't execute that command after multiple attempts."

                time.sleep(0.5)  # Brief pause before retry

    def execute_command(self, text: str) -> str:
        """Execute a recognized voice command"""
        try:
            # Simple command mapping
            if "forward" in text.lower():
                self.move_forward()
                return "Moving forward as requested."
            elif "backward" in text.lower():
                self.move_backward()
                return "Moving backward as requested."
            elif "left" in text.lower():
                self.turn_left()
                return "Turning left as requested."
            elif "right" in text.lower():
                self.turn_right()
                return "Turning right as requested."
            elif "stop" in text.lower():
                self.stop_robot()
                return "Stopping as requested."
            elif "hello" in text.lower() or "hi" in text.lower():
                return "Hello! How can I help you?"
            elif "status" in text.lower():
                return self.get_system_status()
            else:
                return f"I'm not sure how to handle '{text}'. Can you rephrase that?"

        except Exception as e:
            self.get_logger().error(f"Error executing command: {e}")
            raise  # Re-raise to trigger retry logic

    def get_system_status(self) -> str:
        """Get system status for real-world deployment"""
        status = f"System status: {self.performance_metrics['successful_commands']}/{self.performance_metrics['total_commands']} commands successful."
        status += f" Avg confidence: {self.performance_metrics['avg_confidence']:.2f}."
        status += f" Avg processing time: {self.performance_metrics['avg_recognition_time']:.2f}s."
        return status

    def move_forward(self):
        """Move robot forward"""
        msg = Twist()
        msg.linear.x = 0.2  # m/s
        self.cmd_vel_pub.publish(msg)

    def move_backward(self):
        """Move robot backward"""
        msg = Twist()
        msg.linear.x = -0.2  # m/s
        self.cmd_vel_pub.publish(msg)

    def turn_left(self):
        """Turn robot left"""
        msg = Twist()
        msg.angular.z = 0.5  # rad/s
        self.cmd_vel_pub.publish(msg)

    def turn_right(self):
        """Turn robot right"""
        msg = Twist()
        msg.angular.z = -0.5  # rad/s
        self.cmd_vel_pub.publish(msg)

    def stop_robot(self):
        """Stop robot movement"""
        msg = Twist()
        self.cmd_vel_pub.publish(msg)

    def speak(self, text: str):
        """Publish text for robot to speak"""
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)

    def publish_metrics(self):
        """Publish performance metrics"""
        try:
            metrics_msg = String()
            metrics_msg.data = json.dumps(self.performance_metrics)
            self.metrics_pub.publish(metrics_msg)

            # Log metrics if enabled
            if self.logger:
                self.logger.info(f"Performance metrics: {self.performance_metrics}")

        except Exception as e:
            self.get_logger().error(f"Error publishing metrics: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = RealWorldDeploymentNode()

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

## Performance Optimization Examples

This section demonstrates techniques for optimizing voice command system performance:

```python
#!/usr/bin/env python3
"""
Performance Optimization Examples

This example demonstrates various techniques for optimizing
voice command system performance in resource-constrained environments.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import AudioData
from geometry_msgs.msg import Twist
import whisper
import numpy as np
import threading
import queue
import time
import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any
import collections
import gc


@dataclass
class AudioBuffer:
    """Optimized audio buffer for real-time processing"""
    max_duration: float = 3.0  # seconds
    sample_rate: int = 16000
    dtype: np.dtype = np.float32

    def __post_init__(self):
        self.max_samples = int(self.max_duration * self.sample_rate)
        self.buffer = collections.deque(maxlen=self.max_samples)

    def add_audio(self, audio_chunk: np.ndarray):
        """Add audio chunk to buffer"""
        if audio_chunk.dtype != self.dtype:
            audio_chunk = audio_chunk.astype(self.dtype)

        for sample in audio_chunk:
            self.buffer.append(sample)

    def get_audio(self, duration: float = None) -> np.ndarray:
        """Get audio from buffer"""
        if duration:
            samples_needed = int(duration * self.sample_rate)
            start_idx = max(0, len(self.buffer) - samples_needed)
            return np.array(list(self.buffer)[start_idx:], dtype=self.dtype)
        else:
            return np.array(list(self.buffer), dtype=self.dtype)

    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()


class OptimizedVoiceCommandNode(Node):
    def __init__(self):
        super().__init__('optimized_voice_command_node')

        # Use smaller model for better performance
        self.model_size = "base"  # Changed from "small" for better performance
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize Whisper model
        self.get_logger().info(f"Loading Whisper model: {self.model_size} on {self.device}")
        self.whisper_model = whisper.load_model(self.model_size).to(self.device)

        # Initialize optimized audio buffer
        self.audio_buffer = AudioBuffer(max_duration=2.0)  # Reduced for faster processing

        # Command queue
        self.command_queue = queue.Queue()

        # Create publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.speech_pub = self.create_publisher(String, '/robot_speech', 10)

        # Create subscribers
        self.audio_sub = self.create_subscription(
            AudioData,
            'audio_input',
            self.audio_callback,
            10
        )

        # Timer for periodic processing (instead of processing every chunk)
        self.processing_timer = self.create_timer(0.5, self.process_buffered_audio)

        self.get_logger().info("Optimized Voice Command Node initialized")

        # Start command processing thread
        self.command_thread = threading.Thread(target=self.process_commands)
        self.command_thread.daemon = True
        self.command_thread.start()

    def audio_callback(self, msg):
        """Buffer incoming audio instead of processing immediately"""
        try:
            # Convert audio data to numpy array
            audio_array = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0

            # Add to buffer instead of processing immediately
            self.audio_buffer.add_audio(audio_array)

        except Exception as e:
            self.get_logger().error(f"Error buffering audio: {e}")

    def process_buffered_audio(self):
        """Process buffered audio periodically"""
        try:
            # Get accumulated audio from buffer
            audio_data = self.audio_buffer.get_audio(duration=2.0)  # 2 seconds of audio

            if len(audio_data) > 0:
                # Transcribe audio using Whisper
                result = self.whisper_model.transcribe(audio_data, language="en", without_timestamps=True)
                recognized_text = result["text"].strip()

                if recognized_text:
                    self.get_logger().info(f"Recognized: {recognized_text}")

                    # Add to command queue for processing
                    self.command_queue.put(recognized_text)

                # Clear buffer after processing to avoid accumulation
                self.audio_buffer.clear()

        except Exception as e:
            self.get_logger().error(f"Error processing buffered audio: {e}")

    def process_commands(self):
        """Process commands with performance optimization"""
        while rclpy.ok():
            try:
                # Get command from queue with timeout
                command = self.command_queue.get(timeout=1.0)

                # Process the command
                response = self.execute_command_optimized(command)

                if response:
                    self.speak(response)

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Error processing command: {e}")

    def execute_command_optimized(self, text: str) -> str:
        """Execute command with performance optimization"""
        try:
            # Use fast string operations instead of complex NLP
            text_lower = text.lower()

            # Use dictionary lookup for faster command matching
            command_map = {
                "forward": (self.move_forward, "Moving forward as requested."),
                "backward": (self.move_backward, "Moving backward as requested."),
                "left": (self.turn_left, "Turning left as requested."),
                "right": (self.turn_right, "Turning right as requested."),
                "stop": (self.stop_robot, "Stopping as requested."),
                "hello": (None, "Hello! How can I help you?"),
                "hi": (None, "Hello! How can I help you?")
            }

            for command, (action, response) in command_map.items():
                if command in text_lower:
                    if action:
                        action()
                    return response

            return f"I'm not sure how to handle '{text}'. Can you rephrase that?"

        except Exception as e:
            self.get_logger().error(f"Error executing optimized command: {e}")
            return "Sorry, I encountered an error processing that command"

    def move_forward(self):
        """Move robot forward"""
        msg = Twist()
        msg.linear.x = 0.2  # m/s
        self.cmd_vel_pub.publish(msg)

    def move_backward(self):
        """Move robot backward"""
        msg = Twist()
        msg.linear.x = -0.2  # m/s
        self.cmd_vel_pub.publish(msg)

    def turn_left(self):
        """Turn robot left"""
        msg = Twist()
        msg.angular.z = 0.5  # rad/s
        self.cmd_vel_pub.publish(msg)

    def turn_right(self):
        """Turn robot right"""
        msg = Twist()
        msg.angular.z = -0.5  # rad/s
        self.cmd_vel_pub.publish(msg)

    def stop_robot(self):
        """Stop robot movement"""
        msg = Twist()
        self.cmd_vel_pub.publish(msg)

    def speak(self, text: str):
        """Publish text for robot to speak"""
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OptimizedVoiceCommandNode()

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

## Troubleshooting Common Issues

This section provides examples of how to troubleshoot common issues in voice command systems:

```python
#!/usr/bin/env python3
"""
Troubleshooting Common Issues

This example demonstrates how to identify and resolve common issues
in voice command systems for humanoid robots.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import AudioData
from geometry_msgs.msg import Twist
import whisper
import numpy as np
import threading
import queue
import time
import torch
import pyaudio
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging


class TroubleshootingVoiceCommandNode(Node):
    def __init__(self):
        super().__init__('troubleshooting_voice_command_node')

        # Initialize Whisper model
        self.model_size = "small"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.whisper_model = whisper.load_model(self.model_size).to(self.device)
            self.get_logger().info("Whisper model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load Whisper model: {e}")
            self.whisper_model = None

        # Initialize audio parameters
        self.audio_params = {
            'sample_rate': 16000,
            'channels': 1,
            'format': pyaudio.paInt16,
            'chunk_size': 1024
        }

        # Initialize troubleshooting state
        self.troubleshooting_state = {
            'audio_quality': 'unknown',
            'model_status': 'loaded' if self.whisper_model else 'failed',
            'last_error': None,
            'error_count': 0
        }

        # Command queue
        self.command_queue = queue.Queue()

        # Create publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.speech_pub = self.create_publisher(String, '/robot_speech', 10)
        self.diagnostic_pub = self.create_publisher(String, '/diagnostics', 10)

        # Create subscribers
        self.audio_sub = self.create_subscription(
            AudioData,
            'audio_input',
            self.audio_callback,
            10
        )

        # Timer for diagnostics
        self.diagnostics_timer = self.create_timer(5.0, self.publish_diagnostics)

        self.get_logger().info("Troubleshooting Voice Command Node initialized")

        # Start command processing thread
        self.command_thread = threading.Thread(target=self.process_commands)
        self.command_thread.daemon = True
        self.command_thread.start()

    def audio_callback(self, msg):
        """Process incoming audio with troubleshooting"""
        try:
            # Validate audio format
            audio_array = np.frombuffer(msg.data, dtype=np.int16)

            # Check for common audio issues
            if len(audio_array) == 0:
                self.get_logger().warn("Received empty audio data")
                self.troubleshooting_state['audio_quality'] = 'empty'
                return

            # Check for silent audio
            if np.all(audio_array == 0):
                self.get_logger().info("Received silent audio")
                self.troubleshooting_state['audio_quality'] = 'silent'
                return

            # Check for clipping (values at max/min)
            max_val = np.max(np.abs(audio_array))
            if max_val >= 32760:  # Near max for int16
                self.get_logger().warn("Audio may be clipped")
                self.troubleshooting_state['audio_quality'] = 'clipped'

            # Convert to float32
            audio_array = audio_array.astype(np.float32) / 32768.0

            # Validate audio length
            if len(audio_array) < 16000:  # Less than 1 second at 16kHz
                self.get_logger().info("Short audio detected, padding...")
                padded = np.zeros(16000, dtype=audio_array.dtype)
                padded[:len(audio_array)] = audio_array
                audio_array = padded

            # Transcribe audio using Whisper
            if self.whisper_model:
                result = self.whisper_model.transcribe(audio_array, language="en", without_timestamps=True)
                recognized_text = result["text"].strip()

                if recognized_text:
                    self.get_logger().info(f"Recognized: {recognized_text}")
                    self.troubleshooting_state['audio_quality'] = 'good'

                    # Add to command queue for processing
                    self.command_queue.put(recognized_text)
                else:
                    self.get_logger().info("No speech detected in audio")
                    self.troubleshooting_state['audio_quality'] = 'no_speech'
            else:
                self.get_logger().error("Whisper model not loaded")
                self.troubleshooting_state['model_status'] = 'failed'

        except Exception as e:
            self.get_logger().error(f"Error processing audio: {e}")
            self.troubleshooting_state['last_error'] = str(e)
            self.troubleshooting_state['error_count'] += 1

    def process_commands(self):
        """Process commands with error handling"""
        while rclpy.ok():
            try:
                # Get command from queue with timeout
                command = self.command_queue.get(timeout=1.0)

                # Process the command
                response = self.execute_command_with_error_handling(command)

                if response:
                    self.speak(response)

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Error processing command: {e}")
                self.troubleshooting_state['last_error'] = str(e)
                self.troubleshooting_state['error_count'] += 1

    def execute_command_with_error_handling(self, text: str) -> str:
        """Execute command with comprehensive error handling"""
        try:
            # Simple command mapping
            if "forward" in text.lower():
                self.move_forward()
                return "Moving forward as requested."
            elif "backward" in text.lower():
                self.move_backward()
                return "Moving backward as requested."
            elif "left" in text.lower():
                self.turn_left()
                return "Turning left as requested."
            elif "right" in text.lower():
                self.turn_right()
                return "Turning right as requested."
            elif "stop" in text.lower():
                self.stop_robot()
                return "Stopping as requested."
            elif "hello" in text.lower() or "hi" in text.lower():
                return "Hello! How can I help you?"
            elif "diagnostics" in text.lower() or "status" in text.lower():
                return self.get_diagnostics_report()
            elif "troubleshoot" in text.lower():
                return self.run_troubleshooting()
            else:
                return f"I'm not sure how to handle '{text}'. Can you rephrase that?"

        except Exception as e:
            self.get_logger().error(f"Error executing command: {e}")
            self.troubleshooting_state['last_error'] = str(e)
            self.troubleshooting_state['error_count'] += 1
            return "Sorry, I encountered an error processing that command"

    def get_diagnostics_report(self) -> str:
        """Get comprehensive diagnostics report"""
        report = "System diagnostics:\n"
        report += f"- Model status: {self.troubleshooting_state['model_status']}\n"
        report += f"- Audio quality: {self.troubleshooting_state['audio_quality']}\n"
        report += f"- Error count: {self.troubleshooting_state['error_count']}\n"

        if self.troubleshooting_state['last_error']:
            report += f"- Last error: {self.troubleshooting_state['last_error']}\n"

        return report

    def run_troubleshooting(self) -> str:
        """Run comprehensive troubleshooting"""
        issues = []

        # Check model
        if not self.whisper_model:
            issues.append("Whisper model failed to load")

        # Check audio quality
        if self.troubleshooting_state['audio_quality'] in ['silent', 'empty']:
            issues.append("Audio input appears to be silent or empty - check microphone connection")
        elif self.troubleshooting_state['audio_quality'] == 'clipped':
            issues.append("Audio is clipping - reduce microphone gain")

        # Check error count
        if self.troubleshooting_state['error_count'] > 10:
            issues.append("High error count - system may need restart")

        if issues:
            troubleshooting_steps = "\n".join([f"- {issue}" for issue in issues])
            return f"Troubleshooting detected issues:\n{troubleshooting_steps}\nPlease address these issues and try again."
        else:
            return "No issues detected. System appears to be functioning normally."

    def move_forward(self):
        """Move robot forward"""
        try:
            msg = Twist()
            msg.linear.x = 0.2  # m/s
            self.cmd_vel_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error moving forward: {e}")
            self.troubleshooting_state['last_error'] = str(e)

    def move_backward(self):
        """Move robot backward"""
        try:
            msg = Twist()
            msg.linear.x = -0.2  # m/s
            self.cmd_vel_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error moving backward: {e}")
            self.troubleshooting_state['last_error'] = str(e)

    def turn_left(self):
        """Turn robot left"""
        try:
            msg = Twist()
            msg.angular.z = 0.5  # rad/s
            self.cmd_vel_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error turning left: {e}")
            self.troubleshooting_state['last_error'] = str(e)

    def turn_right(self):
        """Turn robot right"""
        try:
            msg = Twist()
            msg.angular.z = -0.5  # rad/s
            self.cmd_vel_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error turning right: {e}")
            self.troubleshooting_state['last_error'] = str(e)

    def stop_robot(self):
        """Stop robot movement"""
        try:
            msg = Twist()
            self.cmd_vel_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error stopping: {e}")
            self.troubleshooting_state['last_error'] = str(e)

    def speak(self, text: str):
        """Publish text for robot to speak"""
        try:
            msg = String()
            msg.data = text
            self.speech_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error speaking: {e}")
            self.troubleshooting_state['last_error'] = str(e)

    def publish_diagnostics(self):
        """Publish diagnostic information"""
        try:
            diagnostics_msg = String()
            diagnostics_msg.data = str(self.troubleshooting_state)
            self.diagnostic_pub.publish(diagnostics_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing diagnostics: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = TroubleshootingVoiceCommandNode()

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

## Exercises

1. Implement a voice command system that can handle multiple languages using Whisper's multilingual capabilities
2. Create a voice command system that learns from user corrections to improve recognition accuracy
3. Implement a context-aware voice command system that remembers previous interactions
4. Build a voice command system with real-time performance monitoring and optimization
5. Create a multimodal voice command system that combines speech with gesture recognition
6. Implement a safety-enhanced voice command system with emergency stop functionality
7. Build a distributed voice command system that can run across multiple robots
8. Create a voice command system with offline processing capabilities
9. Implement a voice command system that can handle noisy environments
10. Build a voice command system with user authentication and personalized responses

## Next Steps

After completing this chapter, you should have a comprehensive understanding of implementing voice command systems for humanoid robots. The next step is to integrate these voice command systems with other robotic capabilities like navigation, manipulation, and computer vision to create truly intelligent and interactive humanoid robots.