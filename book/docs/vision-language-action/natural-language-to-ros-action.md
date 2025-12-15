# Natural Language to ROS 2 Action Translation

## Overview
This chapter covers the critical process of translating natural language commands into executable ROS 2 actions for humanoid robots. This translation layer serves as the bridge between high-level cognitive planning and low-level robot control, enabling robots to understand and execute complex human instructions.

## Learning Objectives
- Understand the architecture of natural language to action translation systems
- Learn to parse and interpret natural language commands
- Design semantic parsers for robotic command interpretation
- Implement action mapping and parameter extraction
- Create robust error handling and fallback mechanisms
- Build validation systems for action safety
- Optimize translation performance for real-time applications

## Prerequisites
- Understanding of ROS 2 action architecture and message types
- Knowledge of natural language processing concepts
- Experience with cognitive planning systems
- Completed Whisper and LLM cognitive planning chapters

## Table of Contents
1. [Introduction to NL-to-Action Translation](#introduction-to-nl-to-action-translation)
2. [Command Parsing and Interpretation](#command-parsing-and-interpretation)
3. [Semantic Parsing for Robotics](#semantic-parsing-for-robotics)
4. [Action Mapping and Parameter Extraction](#action-mapping-and-parameter-extraction)
5. [ROS 2 Action Integration](#ros-2-action-integration)
6. [Safety and Validation Systems](#safety-and-validation-systems)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting Common Issues](#troubleshooting-common-issues)
9. [Best Practices](#best-practices)

## Introduction to NL-to-Action Translation

### The Translation Pipeline
The natural language to ROS 2 action translation process involves several key stages:

1. **Command Reception**: Receiving processed natural language from cognitive systems
2. **Semantic Parsing**: Converting language into structured semantic representations
3. **Action Mapping**: Mapping semantic concepts to ROS 2 action types
4. **Parameter Extraction**: Extracting specific parameters from the command
5. **Validation**: Ensuring the action is safe and feasible
6. **Execution**: Initiating the ROS 2 action with appropriate parameters

### Architecture Components
A typical NL-to-action system includes:

- **Command Parser**: Interprets natural language into structured commands
- **Action Mapper**: Maps commands to ROS 2 action types
- **Parameter Extractor**: Extracts specific values from commands
- **Validator**: Ensures actions are safe and executable
- **Executor**: Initiates ROS 2 action execution

### Example Translation
```
Input: "Please navigate to the kitchen and bring me a cup"
Output: [
  {action: "NavigateToPose", parameters: {pose: "kitchen_location"}},
  {action: "FindObject", parameters: {object_type: "cup", location: "kitchen"}},
  {action: "GraspObject", parameters: {object: "cup", grasp_type: "top_grasp"}},
  {action: "NavigateToPose", parameters: {pose: "user_location"}}
]
```

## Command Parsing and Interpretation

### Command Structure Analysis
Natural language commands typically follow patterns that can be analyzed for robotic execution:

```python
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ParsedCommand:
    action_type: str
    parameters: Dict[str, any]
    confidence: float
    original_text: str

class CommandParser:
    def __init__(self):
        self.action_patterns = self._define_action_patterns()
        self.location_patterns = self._define_location_patterns()
        self.object_patterns = self._define_object_patterns()

    def _define_action_patterns(self) -> Dict[str, List[str]]:
        """Define patterns for different action types"""
        return {
            'navigate': [
                r'go to (?P<location>.+)',
                r'move to (?P<location>.+)',
                r'walk to (?P<location>.+)',
                r'travel to (?P<location>.+)',
                r'navigate to (?P<location>.+)'
            ],
            'grasp': [
                r'pick up (?P<object>.+)',
                r'grasp (?P<object>.+)',
                r'take (?P<object>.+)',
                r'get (?P<object>.+)',
                r'pick (?P<object>.+)'
            ],
            'place': [
                r'place (?P<object>.+) (?:on|at) (?P<location>.+)',
                r'put (?P<object>.+) (?:on|at) (?P<location>.+)',
                r'drop (?P<object>.+) (?:on|at) (?P<location>.+)'
            ],
            'detect': [
                r'find (?P<object>.+)',
                r'locate (?P<object>.+)',
                r'search for (?P<object>.+)',
                r'look for (?P<object>.+)'
            ],
            'speak': [
                r'say "(?P<text>.+)"',
                r'speak "(?P<text>.+)"',
                r'tell me "(?P<text>.+)"'
            ]
        }

    def _define_location_patterns(self) -> List[str]:
        """Define common location identifiers"""
        return [
            r'kitchen', r'living room', r'bedroom', r'bathroom', r'office',
            r'dining room', r'entrance', r'exit', r'corridor', r'hallway',
            r'here', r'there', r'over there', r'nearby'
        ]

    def _define_object_patterns(self) -> List[str]:
        """Define common object types"""
        return [
            r'cup', r'glass', r'bottle', r'book', r'phone', r'keys',
            r'chair', r'table', r'box', r'ball', r'toy', r'food'
        ]

    def parse_command(self, command: str) -> Optional[ParsedCommand]:
        """Parse a natural language command"""
        command = command.lower().strip()

        # Try each action type
        for action_type, patterns in self.action_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, command)
                if match:
                    # Extract parameters from the match
                    parameters = match.groupdict()

                    # Additional parameter processing
                    processed_params = self._process_parameters(parameters, command)

                    return ParsedCommand(
                        action_type=action_type,
                        parameters=processed_params,
                        confidence=self._calculate_confidence(match, command),
                        original_text=command
                    )

        # If no pattern matches, return None
        return None

    def _process_parameters(self, parameters: Dict, command: str) -> Dict:
        """Process and enhance extracted parameters"""
        processed = parameters.copy()

        # Process location parameters
        if 'location' in processed:
            processed['location'] = self._normalize_location(processed['location'])

        # Process object parameters
        if 'object' in processed:
            processed['object'] = self._normalize_object(processed['object'])

        # Add inferred parameters based on context
        processed.update(self._infer_parameters(command))

        return processed

    def _normalize_location(self, location: str) -> str:
        """Normalize location names"""
        # Remove common articles and prepositions
        location = re.sub(r'\b(the|a|an|in|at|on|to|from)\b', '', location)
        location = location.strip()

        # Map common variations to standard names
        location_map = {
            'living room': 'living_room',
            'dining room': 'dining_room',
            'bed room': 'bedroom'
        }

        return location_map.get(location, location)

    def _normalize_object(self, obj: str) -> str:
        """Normalize object names"""
        obj = re.sub(r'\b(the|a|an|my|your|his|her)\b', '', obj)
        return obj.strip()

    def _infer_parameters(self, command: str) -> Dict:
        """Infer additional parameters from context"""
        inferred = {}

        # Infer if the command is urgent
        if any(word in command for word in ['please', 'quickly', 'hurry', 'fast']):
            inferred['priority'] = 'high'
        else:
            inferred['priority'] = 'normal'

        # Infer politeness level
        if 'please' in command:
            inferred['politeness'] = 'polite'
        elif command.endswith('?'):
            inferred['politeness'] = 'polite'
        else:
            inferred['politeness'] = 'neutral'

        return inferred

    def _calculate_confidence(self, match, command: str) -> float:
        """Calculate confidence in the parsing result"""
        # Simple confidence calculation based on match quality
        match_length = len(match.group(0))
        command_length = len(command)

        # Base confidence on how much of the command was matched
        base_confidence = min(1.0, match_length / command_length)

        # Boost confidence if the match covers most of the command
        if match_length / command_length > 0.8:
            base_confidence *= 1.2

        return min(1.0, base_confidence)
```

### Advanced Parsing with Context
Consider context and conversation history for better parsing:

```python
class ContextualCommandParser(CommandParser):
    def __init__(self):
        super().__init__()
        self.conversation_context = []
        self.entity_resolution = EntityResolver()

    def parse_command_with_context(self, command: str, context: Dict) -> Optional[ParsedCommand]:
        """Parse command using additional context information"""
        # Resolve references using context
        resolved_command = self.entity_resolution.resolve_references(command, context)

        # Parse the resolved command
        parsed = self.parse_command(resolved_command)

        if parsed:
            # Enhance with context
            parsed.parameters.update(self._enhance_with_context(parsed.parameters, context))

        return parsed

    def _enhance_with_context(self, parameters: Dict, context: Dict) -> Dict:
        """Enhance parameters with contextual information"""
        enhanced = parameters.copy()

        # If location is ambiguous, use recent context
        if 'location' in enhanced and enhanced['location'] in ['here', 'there']:
            recent_location = context.get('recent_location')
            if recent_location:
                enhanced['location'] = recent_location

        # If object is generic, use recent context
        if 'object' in enhanced and enhanced['object'] in ['it', 'that', 'this']:
            recent_object = context.get('recent_object')
            if recent_object:
                enhanced['object'] = recent_object

        # Add spatial context
        robot_position = context.get('robot_position', {})
        if robot_position:
            enhanced['robot_position'] = robot_position

        return enhanced

class EntityResolver:
    def __init__(self):
        self.definite_articles = ['the', 'this', 'that', 'these', 'those']
        self.demonstratives = ['this', 'that', 'these', 'those']

    def resolve_references(self, command: str, context: Dict) -> str:
        """Resolve pronouns and definite references in the command"""
        # For simplicity, we'll do basic pronoun resolution
        # In practice, this would be more sophisticated

        resolved_command = command.lower()

        # Replace 'it' with the most recently mentioned object
        if 'it' in resolved_command and context.get('recent_object'):
            resolved_command = resolved_command.replace(' it ', f" {context['recent_object']} ")

        # Replace 'there' with the most recently mentioned location
        if 'there' in resolved_command and context.get('recent_location'):
            resolved_command = resolved_command.replace(' there ', f" {context['recent_location']} ")

        return resolved_command
```

## Semantic Parsing for Robotics

### Formal Semantic Representation
Create formal representations of robotic commands:

```python
from enum import Enum
from typing import Union, List
import json

class RobotActionType(Enum):
    NAVIGATE = "navigate"
    GRASP = "grasp"
    PLACE = "place"
    DETECT = "detect"
    SPEAK = "speak"
    GESTURE = "gesture"
    WAIT = "wait"
    QUERY = "query"

class SemanticFrame:
    """Represents the semantic structure of a robotic command"""
    def __init__(self, action_type: RobotActionType,
                 theme: str = None,  # The primary object
                 source: str = None,  # Starting location
                 destination: str = None,  # Target location
                 manner: str = None,  # How to perform the action
                 time: str = None,  # When to perform the action
                 purpose: str = None):  # Why to perform the action
        self.action_type = action_type
        self.theme = theme
        self.source = source
        self.destination = destination
        self.manner = manner
        self.time = time
        self.purpose = purpose

    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            'action_type': self.action_type.value,
            'theme': self.theme,
            'source': self.source,
            'destination': self.destination,
            'manner': self.manner,
            'time': self.time,
            'purpose': self.purpose
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())

class SemanticParser:
    def __init__(self):
        self.verb_action_mapping = {
            'go': RobotActionType.NAVIGATE,
            'move': RobotActionType.NAVIGATE,
            'navigate': RobotActionType.NAVIGATE,
            'walk': RobotActionType.NAVIGATE,
            'pick': RobotActionType.GRASP,
            'grasp': RobotActionType.GRASP,
            'take': RobotActionType.GRASP,
            'get': RobotActionType.GRASP,
            'place': RobotActionType.PLACE,
            'put': RobotActionType.PLACE,
            'drop': RobotActionType.PLACE,
            'find': RobotActionType.DETECT,
            'locate': RobotActionType.DETECT,
            'search': RobotActionType.DETECT,
            'look': RobotActionType.DETECT,
            'say': RobotActionType.SPEAK,
            'speak': RobotActionType.SPEAK,
            'tell': RobotActionType.SPEAK,
            'wave': RobotActionType.GESTURE,
            'point': RobotActionType.GESTURE,
            'wait': RobotActionType.WAIT,
            'stop': RobotActionType.WAIT,
            'how': RobotActionType.QUERY,
            'what': RobotActionType.QUERY,
            'when': RobotActionType.QUERY,
            'where': RobotActionType.QUERY
        }

    def parse_to_semantic_frame(self, command: str) -> SemanticFrame:
        """Parse command into semantic frame"""
        # Simple parsing for demonstration
        words = command.lower().split()

        # Determine action type
        action_type = RobotActionType.NAVIGATE  # Default
        for word in words:
            if word in self.verb_action_mapping:
                action_type = self.verb_action_mapping[word]
                break

        # Extract semantic roles
        theme = self._extract_theme(words, action_type)
        destination = self._extract_location(words, 'destination')
        source = self._extract_location(words, 'source')
        manner = self._extract_manner(words)

        return SemanticFrame(
            action_type=action_type,
            theme=theme,
            source=source,
            destination=destination,
            manner=manner
        )

    def _extract_theme(self, words: List[str], action_type: RobotActionType) -> str:
        """Extract the main object/theme of the action"""
        # Skip common verbs and articles
        skip_words = ['the', 'a', 'an', 'my', 'your', 'and', 'to', 'from', 'at', 'in', 'on']

        for i, word in enumerate(words):
            if word in self.verb_action_mapping or word in skip_words:
                continue

            # If this follows a verb that takes an object
            if i > 0 and words[i-1] in self.verb_action_mapping:
                # Check if it's followed by a preposition (which would indicate location)
                if i + 1 < len(words) and words[i + 1] in ['to', 'at', 'on', 'from']:
                    return word
                elif i + 1 >= len(words):  # Last word
                    return word

        return None

    def _extract_location(self, words: List[str], role: str) -> str:
        """Extract location based on role"""
        # Look for prepositions indicating location
        location_prepositions = {
            'destination': ['to', 'at', 'in', 'on'],
            'source': ['from']
        }

        preps = location_prepositions.get(role, [])

        for i, word in enumerate(words):
            if word in preps and i + 1 < len(words):
                next_word = words[i + 1]
                if next_word not in ['the', 'a', 'an']:
                    return next_word

        return None

    def _extract_manner(self, words: List[str]) -> str:
        """Extract manner adverbs"""
        manner_adverbs = ['slowly', 'quickly', 'carefully', 'gently', 'carefully']

        for word in words:
            if word in manner_adverbs:
                return word

        return None
```

### Ontology-Based Parsing
Use robotics ontologies for more sophisticated parsing:

```python
class RoboticsOntology:
    """Ontology for robotics domain concepts"""

    # Robot capabilities
    capabilities = {
        'navigation': ['navigate', 'move', 'go', 'walk', 'travel'],
        'manipulation': ['grasp', 'pick', 'take', 'place', 'put', 'drop'],
        'perception': ['detect', 'find', 'locate', 'search', 'look'],
        'communication': ['speak', 'say', 'tell', 'greet', 'wave'],
        'locomotion': ['walk', 'move', 'navigate']
    }

    # Objects and categories
    objects = {
        'container': ['cup', 'glass', 'bottle', 'box', 'bowl'],
        'furniture': ['chair', 'table', 'sofa', 'bed', 'desk'],
        'device': ['phone', 'tablet', 'computer', 'lamp'],
        'food': ['apple', 'banana', 'bread', 'water'],
        'personal_item': ['keys', 'wallet', 'book', 'glasses']
    }

    # Locations and spatial relations
    locations = {
        'room': ['kitchen', 'bedroom', 'living room', 'bathroom', 'office'],
        'furniture_location': ['on table', 'in drawer', 'under chair'],
        'spatial_relation': ['on', 'in', 'under', 'next to', 'near', 'far from']
    }

    # Actions and their constraints
    actions_constraints = {
        'grasp': {
            'preconditions': ['object is graspable', 'object is reachable'],
            'effects': ['object is held', 'gripper is occupied']
        },
        'navigate': {
            'preconditions': ['path is clear', 'destination is valid'],
            'effects': ['robot is at destination']
        },
        'place': {
            'preconditions': ['object is held', 'location is suitable'],
            'effects': ['object is placed', 'gripper is free']
        }
    }

class OntologyBasedParser:
    def __init__(self):
        self.ontology = RoboticsOntology()
        self.action_validator = ActionValidator()

    def parse_with_ontology(self, command: str) -> SemanticFrame:
        """Parse command using domain ontology"""
        # First, do basic semantic parsing
        basic_frame = SemanticParser().parse_to_semantic_frame(command)

        # Enhance with ontology knowledge
        enhanced_frame = self._enhance_with_ontology(basic_frame, command)

        return enhanced_frame

    def _enhance_with_ontology(self, frame: SemanticFrame, command: str) -> SemanticFrame:
        """Enhance semantic frame with ontology knowledge"""
        enhanced = frame

        # Categorize theme object
        if frame.theme:
            category = self._categorize_object(frame.theme)
            if category:
                enhanced.theme_category = category

        # Validate action-object compatibility
        if frame.action_type and frame.theme:
            is_compatible = self.action_validator.is_action_object_compatible(
                frame.action_type, frame.theme
            )
            enhanced.action_object_compatible = is_compatible

        # Add spatial constraints
        if frame.destination:
            spatial_constraints = self._get_spatial_constraints(frame.action_type, frame.destination)
            enhanced.spatial_constraints = spatial_constraints

        return enhanced

    def _categorize_object(self, obj: str) -> str:
        """Categorize object based on ontology"""
        for category, objects in self.ontology.objects.items():
            if obj in objects:
                return category
        return 'unknown'

    def _get_spatial_constraints(self, action_type: RobotActionType, location: str) -> Dict:
        """Get spatial constraints for action-location pair"""
        constraints = {}

        if action_type == RobotActionType.GRASP and location in ['under chair', 'in drawer']:
            constraints['reachable'] = False
            constraints['reason'] = 'Object location is not reachable'

        return constraints

class ActionValidator:
    """Validate actions based on domain knowledge"""

    def is_action_object_compatible(self, action_type: RobotActionType, obj: str) -> bool:
        """Check if action is compatible with object"""
        if action_type == RobotActionType.GRASP:
            # Check if object can be grasped
            graspable_objects = (
                self._get_category_objects('container') +
                self._get_category_objects('device') +
                self._get_category_objects('food')
            )
            return obj in graspable_objects

        elif action_type == RobotActionType.PLACE:
            # Check if object can be placed
            return obj not in ['water', 'air', 'smoke']  # Non-placeable objects

        return True

    def _get_category_objects(self, category: str) -> List[str]:
        """Get objects in a specific category from ontology"""
        # This would interface with the actual ontology
        ontology = RoboticsOntology()
        return getattr(ontology, 'objects', {}).get(category, [])
```

## Action Mapping and Parameter Extraction

### ROS 2 Action Mapping
Map semantic frames to specific ROS 2 action types:

```python
import rclpy
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import String
from builtin_interfaces.msg import Duration

class ROS2ActionMapper:
    def __init__(self, node):
        self.node = node
        self.action_clients = {}
        self.parameter_extractors = {
            RobotActionType.NAVIGATE: self._extract_navigate_params,
            RobotActionType.GRASP: self._extract_grasp_params,
            RobotActionType.PLACE: self._extract_place_params,
            RobotActionType.DETECT: self._extract_detect_params,
            RobotActionType.SPEAK: self._extract_speak_params,
        }

    def map_to_ros2_action(self, semantic_frame: SemanticFrame) -> Dict:
        """Map semantic frame to ROS 2 action"""
        action_type = semantic_frame.action_type

        # Get the appropriate parameter extractor
        extractor = self.parameter_extractors.get(action_type)
        if not extractor:
            raise ValueError(f"No parameter extractor for action type: {action_type}")

        # Extract parameters
        params = extractor(semantic_frame)

        # Create ROS 2 action specification
        ros_action = {
            'action_type': self._get_ros2_action_type(action_type),
            'goal': params['goal'],
            'timeout': params.get('timeout', Duration(sec=30, nanosec=0)),
            'feedback_callback': params.get('feedback_callback'),
            'result_callback': params.get('result_callback')
        }

        return ros_action

    def _get_ros2_action_type(self, action_type: RobotActionType):
        """Get the corresponding ROS 2 action type"""
        action_type_mapping = {
            RobotActionType.NAVIGATE: 'nav2_msgs/action/NavigateToPose',
            RobotActionType.GRASP: 'manipulation_msgs/action/GraspObject',
            RobotActionType.PLACE: 'manipulation_msgs/action/PlaceObject',
            RobotActionType.DETECT: 'perception_msgs/action/DetectObjects',
            RobotActionType.SPEAK: 'sound_msgs/action/Speak',
        }

        ros2_type_str = action_type_mapping.get(action_type)
        if not ros2_type_str:
            raise ValueError(f"No ROS 2 action type mapping for: {action_type}")

        # Import the appropriate action type
        module_path, action_name = ros2_type_str.rsplit('/', 1)
        module = __import__(module_path.replace('/', '.'), fromlist=[action_name])
        return getattr(module, action_name)

    def _extract_navigate_params(self, frame: SemanticFrame) -> Dict:
        """Extract parameters for navigation action"""
        # Get destination coordinates from location name
        destination_pose = self._get_location_pose(frame.destination)

        goal = {
            'pose': destination_pose,
            'behavior_tree_id': 'default_nav_tree'
        }

        return {
            'goal': goal,
            'timeout': Duration(sec=60, nanosec=0)  # 1 minute timeout
        }

    def _extract_grasp_params(self, frame: SemanticFrame) -> Dict:
        """Extract parameters for grasping action"""
        goal = {
            'object_name': frame.theme,
            'object_pose': self._get_object_pose(frame.theme),
            'grasp_type': self._determine_grasp_type(frame.theme),
            'gripper_position': 'open'
        }

        return {
            'goal': goal,
            'timeout': Duration(sec=30, nanosec=0)
        }

    def _extract_place_params(self, frame: SemanticFrame) -> Dict:
        """Extract parameters for placing action"""
        goal = {
            'object_name': frame.theme,
            'place_pose': self._get_location_pose(frame.destination),
            'place_surface': self._determine_surface_type(frame.destination)
        }

        return {
            'goal': goal,
            'timeout': Duration(sec=30, nanosec=0)
        }

    def _extract_detect_params(self, frame: SemanticFrame) -> Dict:
        """Extract parameters for detection action"""
        goal = {
            'object_type': frame.theme or 'any',
            'search_area': self._get_search_area(frame.destination),
            'detection_timeout': 10.0
        }

        return {
            'goal': goal,
            'timeout': Duration(sec=15, nanosec=0)
        }

    def _extract_speak_params(self, frame: SemanticFrame) -> Dict:
        """Extract parameters for speaking action"""
        # The text to speak should be in the theme or purpose
        text = frame.theme or frame.purpose or "Hello"

        goal = {
            'text': text,
            'voice_type': 'default',
            'volume': 0.7
        }

        return {
            'goal': goal,
            'timeout': Duration(sec=10, nanosec=0)
        }

    def _get_location_pose(self, location_name: str) -> PoseStamped:
        """Get pose for a named location"""
        # In a real system, this would query a location database
        # For now, we'll use a simple mapping
        location_poses = {
            'kitchen': PoseStamped(pose=Point(x=1.0, y=2.0, z=0.0)),
            'living_room': PoseStamped(pose=Point(x=0.0, y=0.0, z=0.0)),
            'bedroom': PoseStamped(pose=Point(x=-1.0, y=1.0, z=0.0)),
            'office': PoseStamped(pose=Point(x=2.0, y=-1.0, z=0.0)),
        }

        pose = location_poses.get(location_name, PoseStamped(pose=Point(x=0.0, y=0.0, z=0.0)))
        pose.header.frame_id = 'map'
        pose.header.stamp = self.node.get_clock().now().to_msg()

        return pose

    def _get_object_pose(self, object_name: str) -> PoseStamped:
        """Get pose for a named object"""
        # This would normally come from perception system
        # For now, return a default pose
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x = 0.5
        pose.pose.position.y = 0.5
        pose.pose.position.z = 0.0

        return pose

    def _determine_grasp_type(self, object_name: str) -> str:
        """Determine appropriate grasp type for object"""
        if object_name in ['cup', 'glass', 'bottle']:
            return 'top_grasp'
        elif object_name in ['book', 'box']:
            return 'side_grasp'
        else:
            return 'pinch_grasp'

    def _determine_surface_type(self, location_name: str) -> str:
        """Determine surface type for placement"""
        if location_name in ['table', 'desk']:
            return 'horizontal'
        elif location_name in ['shelf']:
            return 'horizontal'
        else:
            return 'any'
```

### Parameter Validation and Refinement
Validate and refine extracted parameters:

```python
class ParameterValidator:
    def __init__(self):
        self.constraints = self._load_constraints()

    def _load_constraints(self) -> Dict:
        """Load parameter constraints for different actions"""
        return {
            'navigate': {
                'max_distance': 10.0,  # meters
                'valid_locations': ['kitchen', 'living_room', 'bedroom', 'office', 'bathroom'],
                'min_time': 1.0,  # seconds
                'max_time': 300.0  # seconds (5 minutes)
            },
            'grasp': {
                'max_object_weight': 2.0,  # kg
                'min_object_size': 0.01,  # meters
                'max_object_size': 0.5,   # meters
                'reachable_distance': 1.0  # meters
            },
            'place': {
                'max_height': 2.0,  # meters
                'min_height': 0.1,  # meters
                'stable_surfaces': ['table', 'counter', 'desk', 'shelf']
            },
            'speak': {
                'max_text_length': 500,  # characters
                'min_volume': 0.1,
                'max_volume': 1.0,
                'valid_voices': ['default', 'male', 'female']
            }
        }

    def validate_parameters(self, action_type: RobotActionType,
                           params: Dict) -> Dict[str, any]:
        """Validate action parameters"""
        action_name = action_type.value
        constraints = self.constraints.get(action_name, {})

        issues = []
        suggestions = []

        # Validate based on action type
        if action_name == 'navigate':
            issues.extend(self._validate_navigate_params(params, constraints))
        elif action_name == 'grasp':
            issues.extend(self._validate_grasp_params(params, constraints))
        elif action_name == 'place':
            issues.extend(self._validate_place_params(params, constraints))
        elif action_name == 'speak':
            issues.extend(self._validate_speak_params(params, constraints))

        # Generate suggestions for invalid parameters
        for issue in issues:
            suggestion = self._generate_suggestion(action_name, issue, params)
            if suggestion:
                suggestions.append(suggestion)

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'suggestions': suggestions,
            'confidence_adjustment': self._calculate_confidence_adjustment(issues)
        }

    def _validate_navigate_params(self, params: Dict, constraints: Dict) -> List[str]:
        """Validate navigation parameters"""
        issues = []

        # Check destination
        destination = params.get('goal', {}).get('pose', {}).get('pose', {}).get('position', {})
        if hasattr(destination, 'x') and hasattr(destination, 'y'):
            distance = (destination.x**2 + destination.y**2)**0.5
            if distance > constraints.get('max_distance', 10.0):
                issues.append(f"Destination too far: {distance:.2f}m > {constraints['max_distance']}m")

        # Check location validity
        location_name = params.get('location_name')
        if location_name and location_name not in constraints.get('valid_locations', []):
            issues.append(f"Invalid location: {location_name}")

        return issues

    def _validate_grasp_params(self, params: Dict, constraints: Dict) -> List[str]:
        """Validate grasping parameters"""
        issues = []

        # Check object properties
        obj_name = params.get('goal', {}).get('object_name', '')
        obj_weight = params.get('object_weight', 0.0)
        obj_size = params.get('object_size', 0.0)

        if obj_weight > constraints.get('max_object_weight', 2.0):
            issues.append(f"Object too heavy: {obj_weight}kg > {constraints['max_object_weight']}kg")

        if obj_size < constraints.get('min_object_size', 0.01):
            issues.append(f"Object too small: {obj_size}m < {constraints['min_object_size']}m")

        if obj_size > constraints.get('max_object_size', 0.5):
            issues.append(f"Object too large: {obj_size}m > {constraints['max_object_size']}m")

        return issues

    def _validate_place_params(self, params: Dict, constraints: Dict) -> List[str]:
        """Validate placement parameters"""
        issues = []

        # Check placement height
        place_pose = params.get('goal', {}).get('place_pose', {})
        if hasattr(place_pose, 'pose') and hasattr(place_pose.pose, 'position'):
            height = place_pose.pose.position.z
            if height > constraints.get('max_height', 2.0):
                issues.append(f"Placement too high: {height}m > {constraints['max_height']}m")
            elif height < constraints.get('min_height', 0.1):
                issues.append(f"Placement too low: {height}m < {constraints['min_height']}m")

        return issues

    def _validate_speak_params(self, params: Dict, constraints: Dict) -> List[str]:
        """Validate speaking parameters"""
        issues = []

        # Check text length
        text = params.get('goal', {}).get('text', '')
        if len(text) > constraints.get('max_text_length', 500):
            issues.append(f"Text too long: {len(text)} chars > {constraints['max_text_length']} chars")

        # Check volume
        volume = params.get('goal', {}).get('volume', 0.7)
        if volume < constraints.get('min_volume', 0.1):
            issues.append(f"Volume too low: {volume} < {constraints['min_volume']}")
        elif volume > constraints.get('max_volume', 1.0):
            issues.append(f"Volume too high: {volume} > {constraints['max_volume']}")

        return issues

    def _generate_suggestion(self, action_name: str, issue: str, params: Dict) -> str:
        """Generate suggestion to fix an issue"""
        if "too far" in issue:
            return "Please specify a closer destination or break the task into steps."
        elif "too heavy" in issue:
            return "Object is too heavy to grasp safely. Consider asking for assistance."
        elif "too high" in issue:
            return "Please specify a lower placement location."
        elif "too long" in issue:
            return "Please shorten your message."
        else:
            return "Please rephrase your command with different parameters."

    def _calculate_confidence_adjustment(self, issues: List[str]) -> float:
        """Calculate confidence adjustment based on issues"""
        if not issues:
            return 0.0

        # More issues = lower confidence
        severity_weights = {
            'critical': -0.4,
            'high': -0.3,
            'medium': -0.2,
            'low': -0.1
        }

        adjustment = 0.0
        for issue in issues:
            # Simple categorization (in practice, use more sophisticated logic)
            if 'too far' in issue or 'too heavy' in issue:
                adjustment += severity_weights['high']
            elif 'too high' in issue or 'too low' in issue:
                adjustment += severity_weights['medium']
            else:
                adjustment += severity_weights['low']

        return max(adjustment, -0.5)  # Don't reduce confidence below 0.5
```

## ROS 2 Action Integration

### Action Execution Manager
Create a comprehensive action execution system:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile
from vision_language_action_msgs.msg import NaturalLanguageCommand, ActionExecutionResult
from vision_language_action_msgs.srv import TranslateCommand
from std_msgs.msg import String
import asyncio
from typing import Dict, List, Optional

class ActionExecutionManager(Node):
    def __init__(self):
        super().__init__('action_execution_manager')

        # Initialize components
        self.ros_mapper = ROS2ActionMapper(self)
        self.validator = ParameterValidator()
        self.parser = ContextualCommandParser()

        # Action clients dictionary
        self.action_clients = {}

        # Publishers and subscribers
        self.result_pub = self.create_publisher(
            ActionExecutionResult, 'action_execution_result', 10
        )
        self.status_pub = self.create_publisher(
            String, 'robot_status', 10
        )

        # Service for command translation
        self.translate_service = self.create_service(
            TranslateCommand, 'translate_command', self.translate_command_callback
        )

        # Context for maintaining state
        self.execution_context = {
            'current_task': None,
            'recent_results': [],
            'robot_state': {},
            'environment_state': {}
        }

        self.get_logger().info("Action execution manager initialized")

    def translate_command_callback(self, request: TranslateCommand.Request,
                                  response: TranslateCommand.Response):
        """Service callback for translating natural language to ROS 2 actions"""
        try:
            # Parse the command
            semantic_frame = self.parser.parse_to_semantic_frame(request.command)

            # Map to ROS 2 action
            ros_action = self.ros_mapper.map_to_ros2_action(semantic_frame)

            # Validate parameters
            validation_result = self.validator.validate_parameters(
                semantic_frame.action_type, ros_action
            )

            if not validation_result['valid']:
                response.success = False
                response.message = f"Validation failed: {validation_result['issues']}"
                response.suggestions = validation_result['suggestions']
                return response

            # Store the action for execution
            action_id = self._generate_action_id()
            self._store_action_for_execution(action_id, ros_action)

            response.success = True
            response.action_id = action_id
            response.message = "Command translated successfully"
            response.estimated_duration = self._estimate_duration(semantic_frame.action_type)

        except Exception as e:
            self.get_logger().error(f"Error translating command: {e}")
            response.success = False
            response.message = str(e)

        return response

    async def execute_action_async(self, action_id: str) -> Dict:
        """Execute a stored action asynchronously"""
        # Retrieve the action
        action = self._retrieve_action(action_id)
        if not action:
            return {'success': False, 'error': f'Action {action_id} not found'}

        # Execute based on action type
        action_type = action['action_type']

        if action_type == 'nav2_msgs/action/NavigateToPose':
            result = await self._execute_navigation_action(action)
        elif action_type == 'manipulation_msgs/action/GraspObject':
            result = await self._execute_manipulation_action(action, 'grasp')
        elif action_type == 'manipulation_msgs/action/PlaceObject':
            result = await self._execute_manipulation_action(action, 'place')
        elif action_type == 'perception_msgs/action/DetectObjects':
            result = await self._execute_perception_action(action)
        elif action_type == 'sound_msgs/action/Speak':
            result = await self._execute_speak_action(action)
        else:
            return {'success': False, 'error': f'Unknown action type: {action_type}'}

        # Update context with result
        self.execution_context['recent_results'].append(result)

        return result

    async def _execute_navigation_action(self, action: Dict) -> Dict:
        """Execute navigation action"""
        try:
            # Create action client if needed
            if 'nav2_msgs/action/NavigateToPose' not in self.action_clients:
                from nav2_msgs.action import NavigateToPose
                self.action_clients['nav2_msgs/action/NavigateToPose'] = ActionClient(
                    self, NavigateToPose, 'navigate_to_pose'
                )

            client = self.action_clients['nav2_msgs/action/NavigateToPose']

            # Wait for server
            client.wait_for_server()

            # Create goal
            from nav2_msgs.action import NavigateToPose
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = action['goal']['pose']

            # Send goal
            goal_future = client.send_goal_async(goal_msg)
            goal_response = await goal_future

            if not goal_response.accepted:
                return {'success': False, 'error': 'Goal rejected'}

            # Get result
            result_future = goal_response.get_result_async()
            result_response = await result_future

            return {
                'success': result_response.result.status == 1,  # SUCCESS
                'result': result_response.result,
                'error': None if result_response.result.status == 1 else 'Navigation failed'
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _execute_manipulation_action(self, action: Dict, action_subtype: str) -> Dict:
        """Execute manipulation action (grasp or place)"""
        try:
            action_msg_type = (
                'manipulation_msgs/action/GraspObject' if action_subtype == 'grasp'
                else 'manipulation_msgs/action/PlaceObject'
            )

            # Create action client if needed
            if action_msg_type not in self.action_clients:
                if action_subtype == 'grasp':
                    from manipulation_msgs.action import GraspObject
                    self.action_clients[action_msg_type] = ActionClient(
                        self, GraspObject, 'grasp_object'
                    )
                else:
                    from manipulation_msgs.action import PlaceObject
                    self.action_clients[action_msg_type] = ActionClient(
                        self, PlaceObject, 'place_object'
                    )

            client = self.action_clients[action_msg_type]
            client.wait_for_server()

            # Create and send goal
            if action_subtype == 'grasp':
                from manipulation_msgs.action import GraspObject
                goal_msg = GraspObject.Goal()
                goal_msg.object_name = action['goal']['object_name']
                goal_msg.grasp_type = action['goal']['grasp_type']
            else:
                from manipulation_msgs.action import PlaceObject
                goal_msg = PlaceObject.Goal()
                goal_msg.object_name = action['goal']['object_name']
                goal_msg.place_pose = action['goal']['place_pose']

            goal_future = client.send_goal_async(goal_msg)
            goal_response = await goal_future

            if not goal_response.accepted:
                return {'success': False, 'error': 'Goal rejected'}

            result_future = goal_response.get_result_async()
            result_response = await result_future

            return {
                'success': result_response.result.success,
                'result': result_response.result,
                'error': None if result_response.result.success else 'Manipulation failed'
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _execute_perception_action(self, action: Dict) -> Dict:
        """Execute perception action"""
        try:
            action_msg_type = 'perception_msgs/action/DetectObjects'

            if action_msg_type not in self.action_clients:
                from perception_msgs.action import DetectObjects
                self.action_clients[action_msg_type] = ActionClient(
                    self, DetectObjects, 'detect_objects'
                )

            client = self.action_clients[action_msg_type]
            client.wait_for_server()

            from perception_msgs.action import DetectObjects
            goal_msg = DetectObjects.Goal()
            goal_msg.object_type = action['goal']['object_type']
            goal_msg.search_area = action['goal']['search_area']

            goal_future = client.send_goal_async(goal_msg)
            goal_response = await goal_future

            if not goal_response.accepted:
                return {'success': False, 'error': 'Goal rejected'}

            result_future = goal_response.get_result_async()
            result_response = await result_future

            return {
                'success': result_response.result.found,
                'result': result_response.result,
                'error': None if result_response.result.found else 'Object not found'
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _execute_speak_action(self, action: Dict) -> Dict:
        """Execute speaking action"""
        try:
            # For speaking, we can use a simple publisher instead of action
            speak_pub = self.create_publisher(String, 'speak_text', 10)

            speak_msg = String()
            speak_msg.data = action['goal']['text']
            speak_pub.publish(speak_msg)

            # Simulate speaking completion
            await asyncio.sleep(len(action['goal']['text']) * 0.1)  # Rough time estimate

            return {
                'success': True,
                'result': {'spoken_text': action['goal']['text']},
                'error': None
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _generate_action_id(self) -> str:
        """Generate unique action ID"""
        import uuid
        return str(uuid.uuid4())

    def _store_action_for_execution(self, action_id: str, action: Dict):
        """Store action for later execution"""
        if not hasattr(self, '_stored_actions'):
            self._stored_actions = {}
        self._stored_actions[action_id] = action

    def _retrieve_action(self, action_id: str) -> Optional[Dict]:
        """Retrieve stored action"""
        if hasattr(self, '_stored_actions'):
            return self._stored_actions.get(action_id)
        return None

    def _estimate_duration(self, action_type: RobotActionType) -> float:
        """Estimate action duration in seconds"""
        duration_map = {
            RobotActionType.NAVIGATE: 30.0,  # 30 seconds for navigation
            RobotActionType.GRASP: 10.0,    # 10 seconds for grasping
            RobotActionType.PLACE: 10.0,    # 10 seconds for placing
            RobotActionType.DETECT: 15.0,   # 15 seconds for detection
            RobotActionType.SPEAK: 5.0,     # 5 seconds for speaking
        }
        return duration_map.get(action_type, 10.0)  # Default 10 seconds

def main(args=None):
    rclpy.init(args=args)
    node = ActionExecutionManager()

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

### Command Sequencing and Planning
Handle complex commands that require multiple actions:

```python
class CommandSequencer:
    def __init__(self, action_executor):
        self.action_executor = action_executor
        self.planning_context = {}

    async def execute_complex_command(self, command_sequence: List[SemanticFrame]) -> Dict:
        """Execute a sequence of commands"""
        results = []
        success = True
        error_message = ""

        for i, frame in enumerate(command_sequence):
            self.action_executor.get_logger().info(f"Executing command {i+1}/{len(command_sequence)}: {frame.action_type}")

            try:
                # Map to ROS 2 action
                ros_action = self.action_executor.ros_mapper.map_to_ros2_action(frame)

                # Validate parameters
                validation_result = self.action_executor.validator.validate_parameters(
                    frame.action_type, ros_action
                )

                if not validation_result['valid']:
                    error_message = f"Validation failed for command {i+1}: {validation_result['issues']}"
                    success = False
                    break

                # Execute action
                result = await self.action_executor.execute_action_async(
                    self.action_executor._generate_action_id()
                )

                results.append({
                    'command_index': i,
                    'action_type': frame.action_type,
                    'result': result,
                    'success': result['success']
                })

                if not result['success']:
                    error_message = f"Command {i+1} failed: {result.get('error', 'Unknown error')}"
                    success = False
                    break

            except Exception as e:
                error_message = f"Error executing command {i+1}: {str(e)}"
                success = False
                break

        return {
            'success': success,
            'results': results,
            'error': error_message if not success else None,
            'total_commands': len(command_sequence),
            'successful_commands': len([r for r in results if r['success']]) if success else i
        }

    def plan_command_sequence(self, high_level_command: str) -> List[SemanticFrame]:
        """Plan a sequence of commands from a high-level command"""
        # This would typically involve:
        # 1. Decomposing the high-level command into subtasks
        # 2. Creating semantic frames for each subtask
        # 3. Determining execution order and dependencies

        # For example: "Go to kitchen and bring me a cup" ->
        # [Navigate to kitchen, Detect cup, Grasp cup, Navigate to user]

        # This is a simplified example - in practice, this would be more sophisticated
        frames = []

        if "kitchen" in high_level_command and ("cup" in high_level_command or "bring" in high_level_command):
            # Navigate to kitchen
            frames.append(SemanticFrame(
                action_type=RobotActionType.NAVIGATE,
                destination="kitchen"
            ))

            # Detect cup
            frames.append(SemanticFrame(
                action_type=RobotActionType.DETECT,
                theme="cup"
            ))

            # Grasp cup
            frames.append(SemanticFrame(
                action_type=RobotActionType.GRASP,
                theme="cup"
            ))

            # Return to user
            frames.append(SemanticFrame(
                action_type=RobotActionType.NAVIGATE,
                destination="user_location"
            ))

        return frames
```

## Safety and Validation Systems

### Comprehensive Safety Framework
Implement safety checks for action execution:

```python
class SafetyFramework:
    def __init__(self):
        self.safety_rules = self._load_safety_rules()
        self.emergency_stop_active = False

    def _load_safety_rules(self) -> Dict:
        """Load comprehensive safety rules"""
        return {
            'navigation': {
                'collision_threshold': 0.5,  # meters
                'speed_limits': {'slow': 0.2, 'normal': 0.5, 'fast': 0.8},  # m/s
                'forbidden_areas': ['exit', 'construction', 'restricted'],
                'minimum_human_distance': 0.8  # meters
            },
            'manipulation': {
                'force_limits': {'min': 5.0, 'max': 50.0},  # Newtons
                'weight_limits': {'min': 0.05, 'max': 2.0},  # kg
                'reachable_distance': 1.2,  # meters
                'forbidden_objects': ['sharp', 'hot', 'fragile', 'liquid']
            },
            'communication': {
                'volume_limits': {'min': 0.1, 'max': 0.8},  # 0-1 scale
                'censored_words': ['offensive', 'inappropriate'],
                'interaction_timeout': 300  # seconds
            }
        }

    def pre_execution_safety_check(self, semantic_frame: SemanticFrame,
                                 robot_state: Dict, environment_state: Dict) -> Dict:
        """Perform safety check before action execution"""
        action_type = semantic_frame.action_type.value

        if action_type == 'navigate':
            return self._check_navigation_safety(semantic_frame, robot_state, environment_state)
        elif action_type == 'grasp':
            return self._check_manipulation_safety(semantic_frame, robot_state, environment_state)
        elif action_type == 'place':
            return self._check_placement_safety(semantic_frame, robot_state, environment_state)
        elif action_type == 'speak':
            return self._check_communication_safety(semantic_frame, robot_state, environment_state)
        else:
            return {'safe': True, 'issues': [], 'risk_level': 'low'}

    def _check_navigation_safety(self, frame: SemanticFrame,
                               robot_state: Dict, env_state: Dict) -> Dict:
        """Check safety for navigation actions"""
        issues = []

        # Check destination validity
        destination = frame.destination
        if destination in self.safety_rules['navigation']['forbidden_areas']:
            issues.append(f"Navigation to forbidden area: {destination}")

        # Check path for obstacles
        path_clear = self._check_navigation_path(robot_state, frame.destination, env_state)
        if not path_clear:
            issues.append("Path to destination is blocked by obstacles")

        # Check distance to humans
        humans_nearby = env_state.get('humans', [])
        robot_pos = robot_state.get('position', {'x': 0, 'y': 0})

        for human in humans_nearby:
            distance = self._calculate_distance(robot_pos, human.get('position', {}))
            if distance < self.safety_rules['navigation']['minimum_human_distance']:
                issues.append(f"Human detected too close: {distance:.2f}m")

        risk_level = 'high' if issues else 'low'
        return {
            'safe': len(issues) == 0,
            'issues': issues,
            'risk_level': risk_level,
            'recommended_action': 'stop' if risk_level == 'high' else 'proceed'
        }

    def _check_manipulation_safety(self, frame: SemanticFrame,
                                 robot_state: Dict, env_state: Dict) -> Dict:
        """Check safety for manipulation actions"""
        issues = []

        # Check object properties
        obj_name = frame.theme
        if obj_name in self.safety_rules['manipulation']['forbidden_objects']:
            issues.append(f"Object is forbidden for manipulation: {obj_name}")

        # Check object weight if known
        obj_weight = env_state.get('objects', {}).get(obj_name, {}).get('weight', 0)
        weight_limits = self.safety_rules['manipulation']['weight_limits']
        if obj_weight > weight_limits['max']:
            issues.append(f"Object too heavy: {obj_weight}kg > {weight_limits['max']}kg")

        # Check reachability
        obj_pos = env_state.get('objects', {}).get(obj_name, {}).get('position', {})
        robot_pos = robot_state.get('position', {'x': 0, 'y': 0})
        distance = self._calculate_distance(robot_pos, obj_pos)

        if distance > self.safety_rules['manipulation']['reachable_distance']:
            issues.append(f"Object not reachable: {distance:.2f}m > {self.safety_rules['manipulation']['reachable_distance']}m")

        risk_level = 'high' if issues else 'low'
        return {
            'safe': len(issues) == 0,
            'issues': issues,
            'risk_level': risk_level,
            'recommended_action': 'stop' if risk_level == 'high' else 'proceed_with_caution'
        }

    def _check_placement_safety(self, frame: SemanticFrame,
                               robot_state: Dict, env_state: Dict) -> Dict:
        """Check safety for placement actions"""
        issues = []

        # Check placement location
        place_location = frame.destination
        if place_location in self.safety_rules['manipulation']['forbidden_objects']:
            issues.append(f"Invalid placement location: {place_location}")

        # Check if placement location is stable
        is_stable = self._check_placement_stability(place_location, env_state)
        if not is_stable:
            issues.append(f"Placement location may not be stable: {place_location}")

        # Check if placement location is occupied
        is_occupied = self._check_placement_occupied(place_location, env_state)
        if is_occupied:
            issues.append(f"Placement location is occupied: {place_location}")

        risk_level = 'high' if issues else 'low'
        return {
            'safe': len(issues) == 0,
            'issues': issues,
            'risk_level': risk_level,
            'recommended_action': 'stop' if risk_level == 'high' else 'proceed'
        }

    def _check_communication_safety(self, frame: SemanticFrame,
                                  robot_state: Dict, env_state: Dict) -> Dict:
        """Check safety for communication actions"""
        issues = []

        # Check for censored words in speech
        text = frame.theme or frame.purpose or ""
        for word in self.safety_rules['communication']['censored_words']:
            if word.lower() in text.lower():
                issues.append(f"Text contains censored word: {word}")

        # Check volume level
        volume = frame.parameters.get('volume', 0.7)
        vol_limits = self.safety_rules['communication']['volume_limits']
        if volume > vol_limits['max']:
            issues.append(f"Volume too loud: {volume} > {vol_limits['max']}")

        risk_level = 'high' if issues else 'low'
        return {
            'safe': len(issues) == 0,
            'issues': issues,
            'risk_level': risk_level,
            'recommended_action': 'stop' if risk_level == 'high' else 'proceed'
        }

    def _check_navigation_path(self, robot_state: Dict, destination: str,
                             env_state: Dict) -> bool:
        """Check if navigation path is clear of obstacles"""
        # This would interface with the navigation system
        # For now, return True (in practice, check costmap, etc.)
        return True

    def _check_placement_stability(self, location: str, env_state: Dict) -> bool:
        """Check if placement location is stable"""
        # Check if it's a known stable surface
        stable_surfaces = ['table', 'counter', 'desk', 'shelf']
        return location in stable_surfaces

    def _check_placement_occupied(self, location: str, env_state: Dict) -> bool:
        """Check if placement location is occupied"""
        # Check environment state for objects at location
        objects = env_state.get('objects', {})
        for obj_name, obj_info in objects.items():
            if obj_info.get('location') == location:
                return True
        return False

    def _calculate_distance(self, pos1: Dict, pos2: Dict) -> float:
        """Calculate Euclidean distance between two positions"""
        dx = pos1.get('x', 0) - pos2.get('x', 0)
        dy = pos1.get('y', 0) - pos2.get('y', 0)
        return (dx**2 + dy**2)**0.5

class SafeActionExecutor:
    def __init__(self, base_executor, safety_framework):
        self.base_executor = base_executor
        self.safety_framework = safety_framework

    async def execute_with_safety(self, semantic_frame: SemanticFrame,
                                robot_state: Dict, environment_state: Dict) -> Dict:
        """Execute action with comprehensive safety checks"""
        # Perform pre-execution safety check
        safety_check = self.safety_framework.pre_execution_safety_check(
            semantic_frame, robot_state, environment_state
        )

        if not safety_check['safe']:
            if safety_check['risk_level'] == 'high':
                return {
                    'success': False,
                    'error': f"Safety check failed: {safety_check['issues']}",
                    'safety_violations': safety_check['issues']
                }
            else:
                # For medium risk, we might want human confirmation
                self.base_executor.get_logger().warning(
                    f"Medium risk action detected: {safety_check['issues']}. Proceeding with caution."
                )

        # Execute the action
        result = await self.base_executor.execute_action_async(
            self.base_executor._generate_action_id()
        )

        # Log the execution for safety monitoring
        self._log_execution(semantic_frame, result, safety_check)

        return result

    def _log_execution(self, frame: SemanticFrame, result: Dict, safety_check: Dict):
        """Log action execution for safety monitoring"""
        # This would typically log to a safety database
        # For now, just log to the console
        self.base_executor.get_logger().info(
            f"Action execution: {frame.action_type.value}, "
            f"Success: {result['success']}, "
            f"Safety: {safety_check['risk_level']}"
        )
```

## Performance Optimization

### Efficient Parsing and Mapping
Optimize the translation pipeline for real-time performance:

```python
import time
from functools import lru_cache
import asyncio
from typing import Optional

class OptimizedNLToActionTranslator:
    def __init__(self):
        self.parser = CommandParser()
        self.ontology_parser = OntologyBasedParser()
        self.ros_mapper = ROS2ActionMapper(None)  # Will be set later
        self.validator = ParameterValidator()

        # Caching for frequently used commands
        self.command_cache = {}
        self.max_cache_size = 1000

        # Pre-compiled regex patterns for faster parsing
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for faster matching"""
        import re

        # Action patterns
        self.compiled_action_patterns = {}
        for action_type, patterns in self.parser.action_patterns.items():
            self.compiled_action_patterns[action_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    @lru_cache(maxsize=128)
    def cached_parse_command(self, command: str) -> Optional[ParsedCommand]:
        """Cached command parsing for frequently used commands"""
        return self.parser.parse_command(command)

    async def translate_command_optimized(self, command: str,
                                        context: Dict = None) -> Dict:
        """Optimized command translation with caching and async processing"""
        start_time = time.time()

        # Check cache first
        cache_key = self._generate_cache_key(command, context)
        cached_result = self.command_cache.get(cache_key)
        if cached_result:
            return cached_result

        # Parse command
        if context:
            parsed = self.ontology_parser.parse_with_ontology(command, context)
        else:
            parsed = self.cached_parse_command(command)

        if not parsed:
            return {
                'success': False,
                'error': 'Could not parse command',
                'processing_time': time.time() - start_time
            }

        # Map to ROS action
        try:
            semantic_frame = self._convert_to_semantic_frame(parsed)
            ros_action = self.ros_mapper.map_to_ros2_action(semantic_frame)
        except Exception as e:
            return {
                'success': False,
                'error': f'Action mapping failed: {str(e)}',
                'processing_time': time.time() - start_time
            }

        # Validate parameters
        validation_result = self.validator.validate_parameters(
            semantic_frame.action_type, ros_action
        )

        result = {
            'success': validation_result['valid'],
            'action': ros_action,
            'semantic_frame': semantic_frame,
            'validation': validation_result,
            'processing_time': time.time() - start_time,
            'command': command
        }

        # Cache result if it's a common command
        if len(self.command_cache) < self.max_cache_size and validation_result['valid']:
            self.command_cache[cache_key] = result

        return result

    def _convert_to_semantic_frame(self, parsed: ParsedCommand) -> SemanticFrame:
        """Convert parsed command to semantic frame"""
        # This would convert the parsed command to a semantic frame
        # For now, create a basic semantic frame
        action_type = RobotActionType[parsed.action_type.upper()] if parsed.action_type.upper() in RobotActionType.__members__ else RobotActionType.NAVIGATE

        return SemanticFrame(
            action_type=action_type,
            theme=parsed.parameters.get('object'),
            destination=parsed.parameters.get('location'),
            purpose=parsed.original_text
        )

    def _generate_cache_key(self, command: str, context: Dict) -> str:
        """Generate cache key for command and context"""
        import hashlib
        cache_input = f"{command}_{str(sorted((context or {}).items()))}"
        return hashlib.md5(cache_input.encode()).hexdigest()

    def cleanup_cache(self):
        """Clean up cache periodically"""
        if len(self.command_cache) > self.max_cache_size * 0.8:
            # Remove oldest entries (in a real system, you'd track access times)
            keys_to_remove = list(self.command_cache.keys())[:len(self.command_cache)//4]
            for key in keys_to_remove:
                del self.command_cache[key]
```

### Parallel Processing
Handle multiple commands efficiently:

```python
import asyncio
import concurrent.futures
from typing import List

class ParallelCommandProcessor:
    def __init__(self, translator, max_workers=4):
        self.translator = translator
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.command_queue = asyncio.Queue()

    async def process_command_batch(self, commands: List[str],
                                  context: Dict = None) -> List[Dict]:
        """Process multiple commands in parallel"""
        tasks = [
            self.translator.translate_command_optimized(cmd, context)
            for cmd in commands
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'success': False,
                    'error': str(result),
                    'command_index': i
                })
            else:
                processed_results.append(result)

        return processed_results

    async def process_streaming_commands(self, command_stream):
        """Process commands as they arrive in a streaming fashion"""
        async for command in command_stream:
            result = await self.translator.translate_command_optimized(command)
            yield result
```

## Troubleshooting Common Issues

### Handling Ambiguous Commands
Implement strategies for dealing with ambiguous inputs:

```python
class AmbiguityHandler:
    def __init__(self):
        self.disambiguation_strategies = {
            'object_reference': self._resolve_object_reference,
            'location_reference': self._resolve_location_reference,
            'action_intent': self._resolve_action_intent,
            'parameter_scope': self._resolve_parameter_scope
        }

    def handle_ambiguous_command(self, command: str, context: Dict) -> Dict:
        """Handle ambiguous commands by requesting clarification or using context"""
        analysis = self._analyze_ambiguity(command, context)

        if not analysis['ambiguous']:
            # Command is clear, return normal translation
            return {'type': 'clear', 'command': command}

        # Determine the best resolution strategy
        resolution = self._apply_resolution_strategy(analysis, context)

        if resolution['type'] == 'resolved':
            return resolution
        elif resolution['type'] == 'request_clarification':
            return resolution
        else:
            # Could not resolve, return best guess
            return {
                'type': 'guess',
                'command': resolution.get('guess', command),
                'confidence': resolution.get('confidence', 0.3),
                'notes': 'Command was ambiguous, using best available interpretation'
            }

    def _analyze_ambiguity(self, command: str, context: Dict) -> Dict:
        """Analyze command for potential ambiguities"""
        analysis = {
            'ambiguous': False,
            'issues': [],
            'types': []
        }

        # Check for pronouns without clear referents
        if any(pronoun in command.lower() for pronoun in ['it', 'that', 'this', 'there']):
            if not context.get('recent_entities'):
                analysis['ambiguous'] = True
                analysis['issues'].append('Pronoun without clear referent')
                analysis['types'].append('reference')

        # Check for generic location references
        if any(placeholder in command.lower() for placeholder in ['there', 'here', 'over there']):
            analysis['ambiguous'] = True
            analysis['issues'].append('Generic location reference')
            analysis['types'].append('location')

        # Check for multiple possible interpretations
        possible_actions = self._find_possible_actions(command)
        if len(possible_actions) > 1:
            analysis['ambiguous'] = True
            analysis['issues'].append(f'Multiple possible actions: {possible_actions}')
            analysis['types'].append('action_intent')

        return analysis

    def _find_possible_actions(self, command: str) -> List[str]:
        """Find multiple possible action interpretations"""
        # This would use more sophisticated NLP in practice
        possible = []
        words = command.lower().split()

        # Look for action words
        action_indicators = {
            'navigation': ['go', 'move', 'navigate', 'walk', 'travel'],
            'manipulation': ['pick', 'grasp', 'take', 'place', 'put'],
            'detection': ['find', 'look', 'search', 'locate'],
            'communication': ['say', 'speak', 'tell', 'ask']
        }

        for action_type, indicators in action_indicators.items():
            if any(indicator in words for indicator in indicators):
                possible.append(action_type)

        return possible

    def _apply_resolution_strategy(self, analysis: Dict, context: Dict) -> Dict:
        """Apply appropriate resolution strategy based on ambiguity type"""
        for ambiguity_type in analysis['types']:
            strategy = self.disambiguation_strategies.get(ambiguity_type)
            if strategy:
                result = strategy(analysis, context)
                if result:
                    return result

        # If no strategy worked, request clarification
        return {
            'type': 'request_clarification',
            'question': self._generate_clarification_question(analysis, context),
            'original_command': analysis.get('original_command', '')
        }

    def _resolve_object_reference(self, analysis: Dict, context: Dict) -> Dict:
        """Resolve ambiguous object references"""
        # Look for the most recently mentioned object
        recent_objects = context.get('recent_objects', [])
        if recent_objects:
            most_recent = recent_objects[-1]  # Most recent object
            resolved_command = analysis.get('original_command', '').replace('it', most_recent)
            return {
                'type': 'resolved',
                'command': resolved_command,
                'resolved_element': 'object_reference',
                'original_reference': 'it',
                'resolved_reference': most_recent
            }

        return None

    def _resolve_location_reference(self, analysis: Dict, context: Dict) -> Dict:
        """Resolve ambiguous location references"""
        # Use the most recent location if available
        recent_location = context.get('recent_location')
        if recent_location:
            resolved_command = analysis.get('original_command', '').replace('there', recent_location)
            return {
                'type': 'resolved',
                'command': resolved_command,
                'resolved_element': 'location_reference',
                'original_reference': 'there',
                'resolved_reference': recent_location
            }

        return None

    def _resolve_action_intent(self, analysis: Dict, context: Dict) -> Dict:
        """Resolve ambiguous action intent"""
        # Use context to determine most likely action
        user_intent = context.get('user_intent')
        if user_intent:
            # Map user intent to likely action
            intent_to_action = {
                'getting_something': 'grasp',
                'moving_somewhere': 'navigate',
                'finding_something': 'detect'
            }

            likely_action = intent_to_action.get(user_intent)
            if likely_action:
                return {
                    'type': 'resolved_with_context',
                    'action_focused_on': likely_action,
                    'confidence': 0.7
                }

        return None

    def _generate_clarification_question(self, analysis: Dict, context: Dict) -> str:
        """Generate appropriate clarification question"""
        if 'reference' in analysis['types']:
            return "Could you please specify what you're referring to?"
        elif 'location' in analysis['types']:
            return "Could you please specify the exact location?"
        elif 'action_intent' in analysis['types']:
            return "Could you clarify what you'd like me to do?"
        else:
            return "I didn't quite understand. Could you please rephrase that?"
```

### Error Recovery and Fallbacks
Implement robust error handling:

```python
class ErrorRecoverySystem:
    def __init__(self):
        self.fallback_strategies = {
            'parsing_failure': self._parsing_fallback,
            'action_mapping_failure': self._action_mapping_fallback,
            'validation_failure': self._validation_fallback,
            'execution_failure': self._execution_fallback
        }

    def handle_translation_error(self, error_type: str, original_command: str,
                               context: Dict = None) -> Dict:
        """Handle different types of translation errors"""
        strategy = self.fallback_strategies.get(error_type)
        if strategy:
            return strategy(original_command, context)
        else:
            return self._default_fallback(original_command, context)

    def _parsing_fallback(self, command: str, context: Dict) -> Dict:
        """Fallback for parsing failures"""
        # Try simpler parsing approaches
        simple_patterns = [
            (r'go.*?(\w+)', 'navigate_to_$1'),
            (r'pick.*?(\w+)', 'grasp_$1'),
            (r'find.*?(\w+)', 'detect_$1'),
        ]

        for pattern, replacement in simple_patterns:
            import re
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                # Create a simpler command that might work
                simplified_command = replacement.replace('$1', match.group(1))
                return {
                    'type': 'simplified_attempt',
                    'simplified_command': simplified_command,
                    'original_command': command,
                    'confidence': 0.4,
                    'notes': 'Original command was difficult to parse, trying simplified version'
                }

        return {
            'type': 'request_repetition',
            'message': 'I couldn\'t understand that command. Could you please repeat it or try rephrasing?',
            'original_command': command
        }

    def _action_mapping_fallback(self, command: str, context: Dict) -> Dict:
        """Fallback for action mapping failures"""
        # Try to map to the most common action types
        common_actions = ['navigate', 'detect', 'speak']

        for action in common_actions:
            if action in command.lower():
                return {
                    'type': 'mapped_to_common',
                    'action': action,
                    'original_command': command,
                    'confidence': 0.5,
                    'notes': 'Used common action mapping as fallback'
                }

        return {
            'type': 'request_clarification',
            'message': 'I understand you want me to do something, but I\'m not sure what action to take. Could you be more specific?',
            'original_command': command
        }

    def _validation_fallback(self, command: str, context: Dict) -> Dict:
        """Fallback for validation failures"""
        # Try to suggest corrections
        suggestions = self._generate_parameter_suggestions(command, context)

        return {
            'type': 'suggest_corrections',
            'original_command': command,
            'suggestions': suggestions,
            'message': 'I understood your command, but some parameters might not be feasible. Here are some suggestions:'
        }

    def _execution_fallback(self, command: str, context: Dict) -> Dict:
        """Fallback for execution failures"""
        return {
            'type': 'alternative_approach',
            'original_command': command,
            'suggestions': [
                'Try breaking the command into smaller steps',
                'Check if the target location/object is accessible',
                'Verify that I have the necessary capabilities for this task'
            ],
            'message': 'I was unable to execute that command. Here are some suggestions:'
        }

    def _default_fallback(self, command: str, context: Dict) -> Dict:
        """Default fallback for unknown error types"""
        return {
            'type': 'general_assistance',
            'message': 'I encountered an issue processing your command. Could you please try rephrasing it?',
            'original_command': command
        }

    def _generate_parameter_suggestions(self, command: str, context: Dict) -> List[str]:
        """Generate parameter suggestions for validation failures"""
        suggestions = []

        # Example suggestions based on common validation issues
        if 'navigate' in command.lower():
            suggestions.append('Try specifying a different destination that is closer.')

        if 'grasp' in command.lower():
            suggestions.append('Make sure the object is within reach and not too heavy.')

        if 'speak' in command.lower():
            suggestions.append('Keep your message shorter for better processing.')

        return suggestions
```

## Best Practices

### 1. Robust Error Handling
Always implement comprehensive error handling and recovery mechanisms:
- Graceful degradation when components fail
- Clear error messages for users
- Fallback strategies for common failure modes
- Logging for debugging and system improvement

### 2. Context Awareness
Maintain and utilize context effectively:
- Conversation history for reference resolution
- Robot state awareness for feasibility checking
- Environmental context for spatial reasoning
- User preferences and interaction patterns

### 3. Safety-First Design
Prioritize safety in all translations:
- Comprehensive safety validation before execution
- Clear safety boundaries and constraints
- Emergency stop capabilities
- Risk assessment for all actions

### 4. Performance Optimization
Optimize for real-time performance:
- Caching for common commands
- Asynchronous processing where possible
- Efficient data structures and algorithms
- Resource management for multi-modal processing

### 5. User Experience
Design for natural and intuitive interactions:
- Clear feedback at each processing step
- Appropriate response times
- Helpful error messages and suggestions
- Consistent interaction patterns

## Exercises

1. Implement a command parser that can handle complex multi-step commands
2. Create a semantic frame system for your specific robot platform
3. Build a ROS 2 action mapper for your robot's capabilities
4. Implement safety validation for each action type
5. Add caching mechanisms to improve response times
6. Create a context management system for conversation history
7. Implement ambiguity resolution strategies for unclear commands
8. Build an error recovery system with appropriate fallbacks
9. Design and implement a command sequencing system for multi-step tasks
10. Create a comprehensive safety framework with multiple validation layers
11. Build a performance monitoring system to track translation efficiency
12. Implement a user feedback system to improve command interpretation over time

## Next Steps

After completing this chapter, you'll understand how to translate natural language commands into executable ROS 2 actions. The next step is to integrate these translation capabilities with your robot's execution system and test the complete pipeline from voice input to action execution.