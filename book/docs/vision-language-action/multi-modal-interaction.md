# Multimodal Interaction Concepts for Humanoid Robots

## Overview
This chapter explores multimodal interaction systems that combine vision, language, and action for humanoid robots. Multimodal interaction enables robots to understand and respond to complex human communication that involves speech, gestures, visual context, and environmental cues simultaneously.

## Learning Objectives
- Understand the principles of multimodal interaction in robotics
- Learn to integrate visual and linguistic information processing
- Design systems that combine multiple sensory inputs
- Implement attention mechanisms for multimodal fusion
- Create context-aware interaction systems
- Evaluate multimodal interaction performance
- Build robust systems that handle ambiguous inputs

## Prerequisites
- Understanding of computer vision and image processing
- Knowledge of natural language processing concepts
- Experience with ROS 2 message types and services
- Completed Whisper and LLM cognitive planning chapters
- Basic understanding of sensor fusion concepts

## Table of Contents
1. [Introduction to Multimodal Interaction](#introduction-to-multimodal-interaction)
2. [Visual-Linguistic Integration](#visual-linguistic-integration)
3. [Attention Mechanisms for Multimodal Fusion](#attention-mechanisms-for-multimodal-fusion)
4. [Gesture Recognition and Interpretation](#gesture-recognition-and-interpretation)
5. [Context-Aware Interaction Systems](#context-aware-interaction-systems)
6. [ROS 2 Integration for Multimodal Systems](#ros-2-integration-for-multimodal-systems)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting Common Issues](#troubleshooting-common-issues)
9. [Best Practices](#best-practices)

## Introduction to Multimodal Interaction

### What is Multimodal Interaction?
Multimodal interaction refers to the ability of a system to process and respond to information from multiple modalities simultaneously. In humanoid robotics, this typically involves:
- **Visual Modality**: Processing images, video, and spatial information
- **Linguistic Modality**: Understanding and generating natural language
- **Gestural Modality**: Recognizing and interpreting human gestures
- **Auditory Modality**: Processing speech and environmental sounds
- **Tactile Modality**: Sensing physical contact and manipulation

### Importance for Humanoid Robots
Humanoid robots benefit significantly from multimodal interaction because:
- **Natural Communication**: Humans naturally communicate using multiple modalities
- **Context Understanding**: Multiple inputs provide richer context for decision-making
- **Robustness**: If one modality fails, others can compensate
- **Ambiguity Resolution**: Multiple cues help resolve ambiguous situations
- **Social Acceptance**: Multimodal behavior appears more natural and approachable

### Challenges in Multimodal Integration
- **Temporal Alignment**: Different modalities may have different processing speeds
- **Feature Fusion**: Combining heterogeneous feature spaces effectively
- **Attention Allocation**: Determining which modalities to prioritize
- **Computational Complexity**: Processing multiple modalities simultaneously
- **Calibration**: Ensuring sensors are properly calibrated and synchronized

## Visual-Linguistic Integration

### Vision-Language Models
Modern vision-language models can understand and generate responses based on both visual and textual inputs:

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
import clip
from typing import Dict, List, Any

class VisionLanguageProcessor:
    def __init__(self, model_name: str = "ViT-B/32"):
        self.model, self.preprocess = clip.load(model_name)
        self.model.eval()

    def encode_image(self, image_path: str) -> torch.Tensor:
        """Encode an image using the vision-language model"""
        image = Image.open(image_path)
        image_input = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
        return image_features

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text using the vision-language model"""
        text_input = clip.tokenize([text])
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
        return text_features

    def compute_similarity(self, image_path: str, texts: List[str]) -> List[float]:
        """Compute similarity between an image and multiple text descriptions"""
        image_features = self.encode_image(image_path)
        text_features = self.encode_text(" ".join(texts))

        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return similarity[0].tolist()

    def find_matching_objects(self, image_path: str, object_names: List[str]) -> Dict[str, float]:
        """Find objects in an image based on name descriptions"""
        results = {}
        for obj_name in object_names:
            # Create descriptive text for the object
            descriptions = [
                f"a photo of {obj_name}",
                f"{obj_name} in the scene",
                f"an image containing {obj_name}"
            ]

            # Compute similarity for each description
            similarities = []
            for desc in descriptions:
                image_features = self.encode_image(image_path)
                text_features = self.encode_text(desc)

                # Normalize and compute similarity
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                similarity = (100.0 * image_features @ text_features.T).item()
                similarities.append(similarity)

            # Use the highest similarity
            results[obj_name] = max(similarities)

        return results
```

### Visual Grounding
Visual grounding connects linguistic descriptions to specific visual elements:

```python
class VisualGroundingSystem:
    def __init__(self):
        # In practice, you would use models like GLIP, GroundingDINO, etc.
        self.object_detector = self._load_object_detector()
        self.text_encoder = self._load_text_encoder()

    def ground_text_in_image(self, image_path: str, text_query: str) -> List[Dict[str, Any]]:
        """Ground a text query in an image to find relevant regions"""
        # Detect objects in the image
        detections = self.object_detector.detect(image_path)

        # Encode the text query
        query_embedding = self.text_encoder.encode(text_query)

        # Match detections to the query
        matches = []
        for detection in detections:
            obj_embedding = self.text_encoder.encode(detection['label'])

            # Compute similarity
            similarity = self._compute_similarity(query_embedding, obj_embedding)

            if similarity > 0.5:  # Threshold for matching
                matches.append({
                    'bbox': detection['bbox'],
                    'label': detection['label'],
                    'confidence': detection['confidence'],
                    'similarity': similarity
                })

        # Sort by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches

    def _compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity between embeddings"""
        return torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()
```

### Scene Understanding
Combine visual and linguistic information for comprehensive scene understanding:

```python
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class SceneObject:
    name: str
    bbox: List[float]  # [x, y, width, height]
    confidence: float
    attributes: List[str]

@dataclass
class SpatialRelation:
    subject: str
    relation: str  # "left of", "right of", "on", "under", etc.
    object: str
    confidence: float

class SceneUnderstandingSystem:
    def __init__(self):
        self.object_detector = self._load_object_detector()
        self.spatial_reasoner = self._load_spatial_reasoner()
        self.language_interpreter = self._load_language_interpreter()

    def understand_scene(self, image_path: str) -> Dict[str, Any]:
        """Comprehensively understand a scene with objects, relations, and context"""
        # Detect objects in the image
        objects = self._detect_objects(image_path)

        # Analyze spatial relations
        relations = self._analyze_spatial_relations(objects)

        # Generate scene description
        description = self._generate_scene_description(objects, relations)

        return {
            'objects': objects,
            'relations': relations,
            'description': description,
            'spatial_layout': self._create_spatial_layout(objects, relations)
        }

    def _detect_objects(self, image_path: str) -> List[SceneObject]:
        """Detect and describe objects in the scene"""
        detections = self.object_detector.detect(image_path)
        objects = []

        for detection in detections:
            obj = SceneObject(
                name=detection['label'],
                bbox=detection['bbox'],
                confidence=detection['confidence'],
                attributes=self._analyze_attributes(image_path, detection['bbox'])
            )
            objects.append(obj)

        return objects

    def _analyze_spatial_relations(self, objects: List[SceneObject]) -> List[SpatialRelation]:
        """Analyze spatial relationships between objects"""
        relations = []

        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i != j:
                    relation = self._determine_spatial_relation(obj1, obj2)
                    if relation:
                        relations.append(relation)

        return relations

    def _determine_spatial_relation(self, obj1: SceneObject, obj2: SceneObject) -> SpatialRelation:
        """Determine spatial relationship between two objects"""
        # Calculate relative positions
        center1 = self._get_bbox_center(obj1.bbox)
        center2 = self._get_bbox_center(obj2.bbox)

        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]

        # Determine relation based on relative positions
        if abs(dx) > abs(dy):  # Horizontal relationship is stronger
            if dx > 0:
                relation = "right of"
            else:
                relation = "left of"
        else:  # Vertical relationship is stronger
            if dy > 0:
                relation = "below"
            else:
                relation = "above"

        # Calculate confidence based on distance and overlap
        distance = (dx**2 + dy**2)**0.5
        confidence = max(0.0, 1.0 - distance / 1000)  # Simplified confidence calculation

        return SpatialRelation(
            subject=obj1.name,
            relation=relation,
            object=obj2.name,
            confidence=confidence
        )

    def _get_bbox_center(self, bbox: List[float]) -> tuple:
        """Get center coordinates of a bounding box"""
        x, y, w, h = bbox
        return (x + w/2, y + h/2)

    def _generate_scene_description(self, objects: List[SceneObject],
                                   relations: List[SpatialRelation]) -> str:
        """Generate a natural language description of the scene"""
        # This would use an LLM or template-based approach
        # For simplicity, we'll create a basic description
        object_names = [obj.name for obj in objects]
        unique_objects = list(set(object_names))

        description = f"The scene contains: {', '.join(unique_objects)}. "

        # Add some spatial relationships
        if relations:
            top_relations = sorted(relations, key=lambda r: r.confidence, reverse=True)[:3]
            relation_descriptions = []
            for rel in top_relations:
                relation_descriptions.append(f"{rel.subject} is {rel.relation} {rel.object}")

            description += "Spatially: " + "; ".join(relation_descriptions) + "."

        return description
```

## Attention Mechanisms for Multimodal Fusion

### Cross-Modal Attention
Implement attention mechanisms that allow different modalities to attend to each other:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == hidden_dim, "Hidden dim must be divisible by num heads"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        query: [batch_size, seq_len_q, hidden_dim] (e.g., text features)
        key: [batch_size, seq_len_k, hidden_dim] (e.g., image features)
        value: [batch_size, seq_len_k, hidden_dim] (e.g., image features)
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k, _ = key.shape[:2]

        # Project to query, key, value
        Q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.hidden_dim
        )

        # Output projection
        output = self.out_proj(attended_values)
        return output, attention_weights

class MultimodalFusionLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.text_to_image_attention = CrossModalAttention(hidden_dim, num_heads)
        self.image_to_text_attention = CrossModalAttention(hidden_dim, num_heads)

        # Layer normalization and feed-forward networks
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.image_norm = nn.LayerNorm(hidden_dim)
        self.text_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.image_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor) -> tuple:
        """
        Fuse text and image features using cross-modal attention
        """
        # Text attending to image features
        attended_text, text_attention = self.text_to_image_attention(
            query=text_features, key=image_features, value=image_features
        )

        # Image attending to text features
        attended_image, image_attention = self.image_to_text_attention(
            query=image_features, key=text_features, value=text_features
        )

        # Residual connections and layer normalization
        fused_text = self.text_norm(text_features + attended_text)
        fused_image = self.image_norm(image_features + attended_image)

        # Feed-forward networks
        fused_text = self.text_norm(fused_text + self.text_ffn(fused_text))
        fused_image = self.image_norm(fused_image + self.image_ffn(fused_image))

        return fused_text, fused_image, text_attention, image_attention
```

### Multimodal Transformer
Build a transformer that processes multiple modalities simultaneously:

```python
class MultimodalTransformer(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 6, num_heads: int = 8):
        super().__init__()
        self.num_layers = num_layers

        # Separate encoders for different modalities
        self.text_encoder = nn.Linear(768, hidden_dim)  # Assuming CLIP text features
        self.image_encoder = nn.Linear(512, hidden_dim)  # Assuming CLIP image features

        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            MultimodalFusionLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Output heads for different tasks
        self.classification_head = nn.Linear(hidden_dim * 2, 2)  # Combined features
        self.generation_head = nn.Linear(hidden_dim, 50257)  # GPT-2 vocab size

    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode features to common dimension
        encoded_text = self.text_encoder(text_features)
        encoded_image = self.image_encoder(image_features)

        # Apply fusion layers
        text_features = encoded_text
        image_features = encoded_image

        attention_weights = []
        for layer in self.fusion_layers:
            text_features, image_features, text_attn, image_attn = layer(text_features, image_features)
            attention_weights.append((text_attn, image_attn))

        # Combine features for output
        combined_features = torch.cat([text_features, image_features], dim=-1)

        return {
            'combined_features': combined_features,
            'text_features': text_features,
            'image_features': image_features,
            'attention_weights': attention_weights
        }
```

### Adaptive Attention
Implement attention that adapts based on the task or context:

```python
class AdaptiveAttention(nn.Module):
    def __init__(self, hidden_dim: int, task_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task_dim = task_dim

        # Task-specific attention parameters
        self.task_embedding = nn.Embedding(10, task_dim)  # 10 different tasks
        self.attention_controller = nn.Sequential(
            nn.Linear(hidden_dim + task_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # For text and image attention weights
            nn.Softmax(dim=-1)
        )

    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor,
                task_id: int) -> tuple:
        # Get task embedding
        task_emb = self.task_embedding(torch.tensor([task_id]))

        # Compute attention weights based on task
        combined_input = torch.cat([
            torch.mean(text_features, dim=1),  # Average over sequence
            task_emb
        ], dim=-1)

        attention_weights = self.attention_controller(combined_input)
        text_weight, image_weight = attention_weights[0]

        # Apply weighted combination
        weighted_text = text_features * text_weight
        weighted_image = image_features * image_weight

        return weighted_text, weighted_image, (text_weight, image_weight)
```

## Gesture Recognition and Interpretation

### Human Pose Estimation
Use pose estimation to recognize and interpret human gestures:

```python
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Dict

class GestureRecognitionSystem:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.gesture_classifier = self._load_gesture_classifier()

    def recognize_gestures(self, image: np.ndarray) -> Dict[str, Any]:
        """Recognize gestures from an image"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        pose_results = self.pose.process(image_rgb)
        hand_results = self.hands.process(image_rgb)

        # Extract pose landmarks
        pose_landmarks = []
        if pose_results.pose_landmarks:
            for landmark in pose_results.pose_landmarks.landmark:
                pose_landmarks.append([landmark.x, landmark.y, landmark.z])

        # Extract hand landmarks
        hand_landmarks = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks_single in hand_results.multi_hand_landmarks:
                hand_landmarks.append([
                    [landmark.x, landmark.y, landmark.z]
                    for landmark in hand_landmarks_single.landmark
                ])

        # Classify gestures
        recognized_gestures = []
        if hand_landmarks:
            for i, hand in enumerate(hand_landmarks):
                gesture = self._classify_hand_gesture(hand)
                if gesture:
                    recognized_gestures.append({
                        'type': 'hand',
                        'gesture': gesture,
                        'hand_index': i,
                        'confidence': gesture['confidence']
                    })

        if pose_landmarks:
            body_gesture = self._classify_body_gesture(pose_landmarks)
            if body_gesture:
                recognized_gestures.append({
                    'type': 'body',
                    'gesture': body_gesture,
                    'confidence': body_gesture['confidence']
                })

        return {
            'gestures': recognized_gestures,
            'pose_landmarks': pose_landmarks,
            'hand_landmarks': hand_landmarks
        }

    def _classify_hand_gesture(self, landmarks: List[List[float]]) -> Dict[str, Any]:
        """Classify hand gesture based on landmarks"""
        # Calculate finger angles
        finger_angles = self._calculate_finger_angles(landmarks)

        # Define gesture patterns
        gesture_patterns = {
            'open_hand': {'thumb': (0, 45), 'fingers': (0, 45)},
            'fist': {'thumb': (135, 180), 'fingers': (135, 180)},
            'pointing': {'thumb': (135, 180), 'index': (0, 45), 'others': (135, 180)},
            'peace': {'thumb': (135, 180), 'index': (0, 45), 'middle': (0, 45), 'others': (135, 180)},
        }

        # Match against patterns
        best_match = None
        best_confidence = 0

        for gesture_name, pattern in gesture_patterns.items():
            confidence = self._match_gesture_pattern(finger_angles, pattern)
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = gesture_name

        if best_match:
            return {
                'name': best_match,
                'confidence': best_confidence,
                'angles': finger_angles
            }

        return None

    def _calculate_finger_angles(self, landmarks: List[List[float]]) -> Dict[str, float]:
        """Calculate angles for each finger"""
        angles = {}

        # Thumb angle (between thumb tip and MCP joint)
        if len(landmarks) > 4:
            thumb_angle = self._calculate_angle(
                landmarks[2][:2],  # MCP joint
                landmarks[3][:2],  # IP joint
                landmarks[4][:2]   # Tip
            )
            angles['thumb'] = thumb_angle

        # Other fingers
        finger_indices = [
            ([5, 6, 7, 8], 'index'),
            ([9, 10, 11, 12], 'middle'),
            ([13, 14, 15, 16], 'ring'),
            ([17, 18, 19, 20], 'pinky')
        ]

        for indices, name in finger_indices:
            if all(i < len(landmarks) for i in indices):
                angle = self._calculate_angle(
                    landmarks[indices[0]][:2],  # MCP
                    landmarks[indices[1]][:2],  # PIP
                    landmarks[indices[3]][:2]   # Tip
                )
                angles[name] = angle

        return angles

    def _calculate_angle(self, p1: List[float], p2: List[float], p3: List[float]) -> float:
        """Calculate angle between three points"""
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        return np.degrees(angle)

    def _match_gesture_pattern(self, angles: Dict[str, float], pattern: Dict) -> float:
        """Match calculated angles against gesture pattern"""
        matches = 0
        total_checks = 0

        for finger, (min_angle, max_angle) in pattern.items():
            if finger in angles:
                total_checks += 1
                if min_angle <= angles[finger] <= max_angle:
                    matches += 1

        return matches / total_checks if total_checks > 0 else 0

    def _classify_body_gesture(self, landmarks: List[List[float]]) -> Dict[str, Any]:
        """Classify body gestures based on pose landmarks"""
        # Calculate body angles and positions
        if len(landmarks) < 25:
            return None

        # Example: Wave gesture detection
        left_wrist = np.array(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value][:2])
        left_shoulder = np.array(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value][:2])
        right_wrist = np.array(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value][:2])
        right_shoulder = np.array(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value][:2])

        # Check if arms are raised (like waving)
        left_arm_raised = left_wrist[1] < left_shoulder[1] + 0.1
        right_arm_raised = right_wrist[1] < right_shoulder[1] + 0.1

        if left_arm_raised or right_arm_raised:
            return {
                'name': 'wave',
                'confidence': 0.8 if left_arm_raised and right_arm_raised else 0.6,
                'arms_raised': {'left': left_arm_raised, 'right': right_arm_raised}
            }

        return None
```

### Gesture-to-Command Mapping
Map recognized gestures to robot commands:

```python
class GestureCommandMapper:
    def __init__(self):
        self.gesture_commands = {
            'open_hand': {'action': 'stop', 'priority': 'medium'},
            'fist': {'action': 'grasp', 'priority': 'high'},
            'pointing': {'action': 'navigate_to_pointed_location', 'priority': 'high'},
            'peace': {'action': 'take_photo', 'priority': 'medium'},
            'wave': {'action': 'greet', 'priority': 'low'},
        }

        self.gesture_contexts = {
            'navigation_mode': ['pointing'],
            'manipulation_mode': ['open_hand', 'fist'],
            'interaction_mode': ['wave', 'peace']
        }

    def map_gesture_to_command(self, gesture: Dict, context: str = 'general') -> Dict[str, Any]:
        """Map a recognized gesture to a robot command"""
        gesture_name = gesture['name']

        # Check if gesture is valid in current context
        valid_gestures = self.gesture_contexts.get(context, list(self.gesture_commands.keys()))
        if gesture_name not in valid_gestures:
            return {
                'action': 'ignore',
                'reason': f'Gesture {gesture_name} not valid in {context} context',
                'confidence': 0.0
            }

        # Get command mapping
        command_mapping = self.gesture_commands.get(gesture_name)
        if not command_mapping:
            return {
                'action': 'unknown',
                'reason': f'No command mapping for gesture {gesture_name}',
                'confidence': 0.0
            }

        # Adjust confidence based on gesture confidence
        adjusted_confidence = min(1.0, gesture['confidence'] * 1.2)  # Boost slightly

        return {
            'action': command_mapping['action'],
            'priority': command_mapping['priority'],
            'confidence': adjusted_confidence,
            'parameters': self._extract_parameters(gesture, command_mapping['action'])
        }

    def _extract_parameters(self, gesture: Dict, action: str) -> Dict[str, Any]:
        """Extract action-specific parameters from gesture"""
        params = {'gesture_data': gesture}

        if action == 'navigate_to_pointed_location':
            # Extract pointing direction from hand landmarks
            hand_landmarks = gesture.get('hand_landmarks', [])
            if hand_landmarks:
                # Calculate direction vector from shoulder to hand
                # This is simplified - in practice you'd need camera calibration
                params['direction'] = self._calculate_pointing_direction(hand_landmarks)

        elif action == 'grasp':
            # For grasping, you might want to know which hand
            params['hand'] = gesture.get('hand_index', 0)

        return params

    def _calculate_pointing_direction(self, hand_landmarks: List) -> Dict[str, float]:
        """Calculate the direction of pointing gesture"""
        # Simplified calculation - assumes 2D image coordinates
        if len(hand_landmarks) > 8:  # Index finger landmarks
            index_finger_tip = hand_landmarks[8][:2]  # Tip of index finger
            index_finger_pip = hand_landmarks[6][:2]  # PIP joint

            direction_vector = [
                index_finger_tip[0] - index_finger_pip[0],
                index_finger_tip[1] - index_finger_pip[1]
            ]

            # Normalize
            length = (direction_vector[0]**2 + direction_vector[1]**2)**0.5
            if length > 0:
                direction_vector = [v/length for v in direction_vector]

            return {
                'x': direction_vector[0],
                'y': direction_vector[1],
                'magnitude': length
            }

        return {'x': 0, 'y': 0, 'magnitude': 0}
```

## Context-Aware Interaction Systems

### Context Modeling
Build systems that maintain and use contextual information:

```python
from datetime import datetime
from typing import Optional
import json

class ContextModel:
    def __init__(self):
        self.current_task = None
        self.user_intent = None
        self.spatial_context = {}
        self.temporal_context = {}
        self.social_context = {}
        self.environment_context = {}
        self.history = []

    def update_context(self, **kwargs):
        """Update various context components"""
        if 'task' in kwargs:
            self.current_task = kwargs['task']
        if 'user_intent' in kwargs:
            self.user_intent = kwargs['user_intent']
        if 'spatial' in kwargs:
            self.spatial_context.update(kwargs['spatial'])
        if 'environment' in kwargs:
            self.environment_context.update(kwargs['environment'])

        # Add to history
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'update': kwargs
        })

    def get_context_relevance(self, modality: str, information: Any) -> float:
        """Determine how relevant information is given the current context"""
        relevance_score = 0.5  # Base score

        # Task relevance
        if self.current_task:
            relevance_score += self._calculate_task_relevance(modality, information)

        # Spatial relevance
        if self.spatial_context:
            relevance_score += self._calculate_spatial_relevance(modality, information)

        # Temporal relevance
        relevance_score += self._calculate_temporal_relevance(modality, information)

        return max(0.0, min(1.0, relevance_score))

    def _calculate_task_relevance(self, modality: str, information: Any) -> float:
        """Calculate relevance based on current task"""
        task_relevance_map = {
            'navigation': {'vision': 0.8, 'language': 0.7, 'gesture': 0.3},
            'manipulation': {'vision': 0.9, 'language': 0.6, 'gesture': 0.4},
            'conversation': {'language': 0.9, 'vision': 0.4, 'gesture': 0.7},
        }

        if self.current_task in task_relevance_map:
            modality_relevance = task_relevance_map[self.current_task].get(modality, 0.1)
            return modality_relevance

        return 0.0

    def _calculate_spatial_relevance(self, modality: str, information: Any) -> float:
        """Calculate relevance based on spatial context"""
        # If information is near the robot's location, increase relevance
        if modality == 'vision' and 'location' in information:
            robot_location = self.spatial_context.get('robot_position', {})
            info_location = information['location']

            distance = self._calculate_distance(robot_location, info_location)
            if distance < 2.0:  # Within 2 meters
                return 0.3
            elif distance < 5.0:  # Within 5 meters
                return 0.1

        return 0.0

    def _calculate_temporal_relevance(self, modality: str, information: Any) -> float:
        """Calculate relevance based on temporal context"""
        # Recent information is more relevant
        current_time = datetime.now()
        if 'timestamp' in information:
            info_time = datetime.fromisoformat(information['timestamp'])
            time_diff = (current_time - info_time).total_seconds()

            if time_diff < 5:  # Less than 5 seconds old
                return 0.3
            elif time_diff < 30:  # Less than 30 seconds old
                return 0.1

        return 0.0

    def _calculate_distance(self, pos1: Dict, pos2: Dict) -> float:
        """Calculate distance between two positions"""
        dx = pos1.get('x', 0) - pos2.get('x', 0)
        dy = pos1.get('y', 0) - pos2.get('y', 0)
        dz = pos1.get('z', 0) - pos2.get('z', 0)
        return (dx**2 + dy**2 + dz**2)**0.5

class ContextAwareInteractionSystem:
    def __init__(self):
        self.context_model = ContextModel()
        self.modality_processors = {
            'vision': self._process_vision_input,
            'language': self._process_language_input,
            'gesture': self._process_gesture_input
        }

    def process_multimodal_input(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process multimodal input with context awareness"""
        processed_outputs = {}

        for modality, data in inputs.items():
            if modality in self.modality_processors:
                # Calculate relevance based on context
                relevance = self.context_model.get_context_relevance(modality, data)

                if relevance > 0.3:  # Process if somewhat relevant
                    processed_output = self.modality_processors[modality](data, relevance)
                    processed_outputs[modality] = processed_output

        # Fuse the processed outputs based on context and relevance
        fused_output = self._fuse_multimodal_outputs(processed_outputs)

        return fused_output

    def _process_vision_input(self, data: Dict, relevance: float) -> Dict:
        """Process visual input with context awareness"""
        return {
            'objects': data.get('objects', []),
            'scene_description': data.get('description', ''),
            'relevance_score': relevance,
            'confidence': data.get('confidence', 0.8) * relevance
        }

    def _process_language_input(self, data: Dict, relevance: float) -> Dict:
        """Process language input with context awareness"""
        return {
            'intent': data.get('intent', ''),
            'entities': data.get('entities', []),
            'relevance_score': relevance,
            'confidence': data.get('confidence', 0.9) * relevance
        }

    def _process_gesture_input(self, data: Dict, relevance: float) -> Dict:
        """Process gesture input with context awareness"""
        return {
            'gesture': data.get('gesture', ''),
            'command': data.get('command', ''),
            'relevance_score': relevance,
            'confidence': data.get('confidence', 0.7) * relevance
        }

    def _fuse_multimodal_outputs(self, processed_outputs: Dict) -> Dict:
        """Fuse outputs from different modalities based on context"""
        # Calculate weighted combination based on relevance and confidence
        total_weight = 0
        weighted_outputs = []

        for modality, output in processed_outputs.items():
            weight = output['relevance_score'] * output['confidence']
            total_weight += weight

            weighted_outputs.append({
                'modality': modality,
                'output': output,
                'weight': weight
            })

        # Normalize weights
        if total_weight > 0:
            for item in weighted_outputs:
                item['normalized_weight'] = item['weight'] / total_weight

        # Generate final response based on weighted inputs
        final_response = {
            'fused_outputs': weighted_outputs,
            'total_confidence': total_weight,
            'primary_modality': self._determine_primary_modality(weighted_outputs),
            'integrated_action': self._integrate_actions(weighted_outputs)
        }

        return final_response

    def _determine_primary_modality(self, weighted_outputs: List[Dict]) -> str:
        """Determine which modality has the highest weight"""
        if not weighted_outputs:
            return 'none'

        primary = max(weighted_outputs, key=lambda x: x['normalized_weight'])
        return primary['modality']

    def _integrate_actions(self, weighted_outputs: List[Dict]) -> Dict:
        """Integrate actions from different modalities"""
        actions = []
        for item in weighted_outputs:
            modality = item['modality']
            output = item['output']

            if 'command' in output or 'intent' in output:
                action = {
                    'modality': modality,
                    'action': output.get('command') or output.get('intent'),
                    'confidence': item['normalized_weight'],
                    'parameters': output.get('entities', {})
                }
                actions.append(action)

        # Sort by confidence and return the highest confidence action
        if actions:
            actions.sort(key=lambda x: x['confidence'], reverse=True)
            return actions[0]

        return {'action': 'none', 'confidence': 0.0}
```

### Attention Allocation
Implement systems that allocate attention based on context:

```python
class AttentionAllocationSystem:
    def __init__(self):
        self.attention_weights = {
            'vision': 0.4,
            'language': 0.4,
            'gesture': 0.2
        }

        self.task_attention_profiles = {
            'navigation': {'vision': 0.6, 'language': 0.3, 'gesture': 0.1},
            'manipulation': {'vision': 0.7, 'language': 0.2, 'gesture': 0.1},
            'conversation': {'language': 0.6, 'vision': 0.3, 'gesture': 0.1},
            'greeting': {'gesture': 0.5, 'language': 0.3, 'vision': 0.2},
        }

    def allocate_attention(self, current_task: str, environmental_factors: Dict) -> Dict[str, float]:
        """Allocate attention weights based on task and environment"""
        # Start with task-specific profile
        if current_task in self.task_attention_profiles:
            attention = self.task_attention_profiles[current_task].copy()
        else:
            attention = self.attention_weights.copy()

        # Adjust based on environmental factors
        if environmental_factors.get('noise_level', 0) > 0.7:
            # In noisy environments, reduce language attention
            attention['language'] *= 0.5
            attention['vision'] += attention['language'] * 0.5
            attention['language'] *= 0.5

        if environmental_factors.get('lighting', 1.0) < 0.3:
            # In low light, reduce vision attention
            vision_reduction = attention['vision'] * 0.3
            attention['vision'] -= vision_reduction
            attention['language'] += vision_reduction * 0.7
            attention['gesture'] += vision_reduction * 0.3

        # Normalize to ensure sum is 1.0
        total = sum(attention.values())
        if total > 0:
            for modality in attention:
                attention[modality] /= total

        return attention

    def update_attention_for_context(self, context: Dict) -> None:
        """Update attention based on current context"""
        # Example: If user is pointing at something, increase vision attention
        if context.get('user_gesture') == 'pointing':
            self.attention_weights['vision'] = min(0.8, self.attention_weights['vision'] + 0.2)
            # Reduce other modalities proportionally
            remaining = 1.0 - self.attention_weights['vision']
            self.attention_weights['language'] = remaining * 0.6
            self.attention_weights['gesture'] = remaining * 0.4
```

## ROS 2 Integration for Multimodal Systems

### Multimodal Perception Node
Create a ROS 2 node that integrates multiple sensory inputs:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from vision_language_action_msgs.msg import MultimodalInput, MultimodalOutput
from vision_language_action_msgs.msg import GestureData, SceneDescription
import cv2
import numpy as np
from cv_bridge import CvBridge
import threading
import queue

class MultimodalPerceptionNode(Node):
    def __init__(self):
        super().__init__('multimodal_perception_node')

        # Initialize components
        self.bridge = CvBridge()
        self.gesture_recognizer = GestureRecognitionSystem()
        self.scene_understander = SceneUnderstandingSystem()
        self.context_aware_system = ContextAwareInteractionSystem()

        # Queues for synchronization
        self.image_queue = queue.Queue(maxsize=5)
        self.language_queue = queue.Queue(maxsize=5)

        # Publishers
        self.output_pub = self.create_publisher(
            MultimodalOutput, 'multimodal_output', 10
        )
        self.scene_pub = self.create_publisher(
            SceneDescription, 'scene_description', 10
        )
        self.gesture_pub = self.create_publisher(
            GestureData, 'recognized_gestures', 10
        )

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback,
            QoSProfile(depth=10)
        )
        self.compressed_image_sub = self.create_subscription(
            CompressedImage, 'camera/image_compressed',
            self.compressed_image_callback, QoSProfile(depth=10)
        )
        self.language_sub = self.create_subscription(
            String, 'recognized_text', self.language_callback,
            QoSProfile(depth=10)
        )

        # Timer for processing multimodal inputs
        self.process_timer = self.create_timer(0.1, self.process_multimodal_inputs)

        self.get_logger().info("Multimodal perception node initialized")

    def image_callback(self, msg: Image):
        """Handle raw image messages"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_queue.put_nowait(cv_image)
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def compressed_image_callback(self, msg: CompressedImage):
        """Handle compressed image messages"""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.image_queue.put_nowait(cv_image)
        except Exception as e:
            self.get_logger().error(f"Error processing compressed image: {e}")

    def language_callback(self, msg: String):
        """Handle language input messages"""
        try:
            self.language_queue.put_nowait(msg.data)
        except Exception as e:
            self.get_logger().error(f"Error processing language: {e}")

    def process_multimodal_inputs(self):
        """Process synchronized multimodal inputs"""
        try:
            # Get latest image
            image = None
            while not self.image_queue.empty():
                image = self.image_queue.get_nowait()

            # Get latest language
            language = None
            while not self.language_queue.empty():
                language = self.language_queue.get_nowait()

            if image is not None or language is not None:
                # Process the inputs
                multimodal_inputs = {}

                if image is not None:
                    # Process visual information
                    gesture_data = self.gesture_recognizer.recognize_gestures(image)
                    scene_data = self.scene_understander.understand_scene_from_image(image)

                    multimodal_inputs['vision'] = {
                        'gestures': gesture_data['gestures'],
                        'scene': scene_data,
                        'timestamp': self.get_clock().now().to_msg()
                    }

                if language is not None:
                    multimodal_inputs['language'] = {
                        'text': language,
                        'timestamp': self.get_clock().now().to_msg()
                    }

                # Process with context awareness
                result = self.context_aware_system.process_multimodal_input(multimodal_inputs)

                # Publish results
                self.publish_multimodal_output(result, multimodal_inputs)

        except Exception as e:
            self.get_logger().error(f"Error in multimodal processing: {e}")

    def publish_multimodal_output(self, result: Dict, inputs: Dict):
        """Publish multimodal processing results"""
        output_msg = MultimodalOutput()
        output_msg.header.stamp = self.get_clock().now().to_msg()

        # Set primary modality
        output_msg.primary_modality = result.get('primary_modality', 'none')

        # Set integrated action
        integrated_action = result.get('integrated_action', {})
        output_msg.action = integrated_action.get('action', 'none')
        output_msg.action_confidence = integrated_action.get('confidence', 0.0)

        # Set fused outputs
        for item in result.get('fused_outputs', []):
            modality_output = MultimodalOutput()
            modality_output.modality = item['modality']
            modality_output.confidence = item['normalized_weight']
            output_msg.modality_outputs.append(modality_output)

        self.output_pub.publish(output_msg)

    def understand_scene_from_image(self, cv_image: np.ndarray) -> Dict:
        """Understand scene from OpenCV image"""
        # Convert to appropriate format for scene understanding
        # This would involve saving temporary file or using in-memory processing
        # For now, we'll simulate the output
        return {
            'objects': ['person', 'chair', 'table'],
            'description': 'A person sitting at a table with a chair',
            'spatial_relations': ['person sitting on chair', 'chair at table']
        }

def main(args=None):
    rclpy.init(args=args)
    node = MultimodalPerceptionNode()

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

### Multimodal Command Execution Node
Create a node that executes commands based on multimodal input:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from vision_language_action_msgs.msg import MultimodalOutput
from vision_language_action_msgs.srv import ExecutePlan
from geometry_msgs.msg import Point
from std_msgs.msg import String

class MultimodalCommandExecutor(Node):
    def __init__(self):
        super().__init__('multimodal_command_executor')

        # Initialize action clients for different robot capabilities
        self.navigation_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.manipulation_client = ActionClient(self, ManipulateObject, 'manipulate_object')

        # Service client for plan execution
        self.plan_executor = self.create_client(ExecutePlan, 'execute_plan')

        # Subscriber for multimodal outputs
        self.multimodal_sub = self.create_subscription(
            MultimodalOutput, 'multimodal_output',
            self.multimodal_callback, 10
        )

        # Publisher for robot status
        self.status_pub = self.create_publisher(String, 'robot_status', 10)

        self.get_logger().info("Multimodal command executor initialized")

    def multimodal_callback(self, msg: MultimodalOutput):
        """Handle multimodal output and execute appropriate command"""
        if msg.action_confidence < 0.5:
            self.get_logger().info(f"Action confidence too low: {msg.action_confidence}")
            return

        action = msg.action
        self.get_logger().info(f"Executing action: {action} with confidence {msg.action_confidence}")

        if action == 'navigate_to_pointed_location':
            self.execute_navigation_command(msg)
        elif action == 'greet':
            self.execute_greeting_command(msg)
        elif action == 'take_photo':
            self.execute_photo_command(msg)
        elif action == 'grasp':
            self.execute_grasp_command(msg)
        else:
            self.get_logger().info(f"Unknown action: {action}")

    def execute_navigation_command(self, msg: MultimodalOutput):
        """Execute navigation based on pointing gesture"""
        # Extract navigation target from multimodal output
        # This would involve processing the pointing direction and converting to coordinates
        target_point = self._extract_navigation_target(msg)

        if target_point:
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose.pose.position = target_point
            goal_msg.pose.header.frame_id = 'map'

            self.navigation_client.wait_for_server()
            future = self.navigation_client.send_goal_async(goal_msg)
            future.add_done_callback(self.navigation_done_callback)

    def execute_greeting_command(self, msg: MultimodalOutput):
        """Execute greeting action"""
        # For greeting, we might want to look at the person, wave, and speak
        self.get_logger().info("Executing greeting command")

        # Look toward the person
        self.look_at_person(msg)

        # Wave gesture
        self.perform_wave()

        # Speak greeting
        self.speak_greeting()

    def execute_photo_command(self, msg: MultimodalOutput):
        """Execute photo capture command"""
        self.get_logger().info("Capturing photo")
        # This would involve triggering the camera system
        # and potentially storing the image with context

    def execute_grasp_command(self, msg: MultimodalOutput):
        """Execute grasping command"""
        self.get_logger().info("Executing grasp command")
        # This would involve identifying an object and planning a grasp

    def _extract_navigation_target(self, msg: MultimodalOutput) -> Point:
        """Extract navigation target from multimodal message"""
        # This would involve processing visual direction information
        # and converting to map coordinates
        # For now, return a dummy point
        point = Point()
        point.x = 1.0
        point.y = 1.0
        point.z = 0.0
        return point

    def look_at_person(self, msg: MultimodalOutput):
        """Make robot look at person"""
        # Implementation would control head/neck joints
        pass

    def perform_wave(self):
        """Perform waving gesture"""
        # Implementation would control arm joints
        pass

    def speak_greeting(self):
        """Speak a greeting"""
        greeting_pub = self.create_publisher(String, 'speak_text', 10)
        greeting_msg = String()
        greeting_msg.data = "Hello! Nice to meet you."
        greeting_pub.publish(greeting_msg)

    def navigation_done_callback(self, future):
        """Handle navigation completion"""
        result = future.result()
        if result.success:
            self.get_logger().info("Navigation completed successfully")
        else:
            self.get_logger().error("Navigation failed")

def main(args=None):
    rclpy.init(args=args)
    node = MultimodalCommandExecutor()

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

## Performance Optimization

### Efficient Multimodal Processing
Optimize processing for real-time performance:

```python
import asyncio
import concurrent.futures
from collections import deque
import time

class EfficientMultimodalProcessor:
    def __init__(self, max_workers: int = 4):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.processing_cache = {}
        self.fps_limiters = {
            'vision': 10,  # 10 FPS for heavy vision processing
            'language': 5,  # 5 FPS for LLM calls
            'gesture': 15   # 15 FPS for gesture recognition
        }
        self.last_process_times = {
            'vision': 0,
            'language': 0,
            'gesture': 0
        }

        # Circular buffers for temporal consistency
        self.temporal_buffers = {
            'gestures': deque(maxlen=5),
            'objects': deque(maxlen=3)
        }

    async def process_multimodal_frame(self, frame_data: Dict) -> Dict:
        """Process a single multimodal frame efficiently"""
        current_time = time.time()
        results = {}

        # Process each modality based on its FPS limit
        for modality, fps_limit in self.fps_limiters.items():
            min_interval = 1.0 / fps_limit
            time_since_last = current_time - self.last_process_times[modality]

            if time_since_last >= min_interval:
                # Process this modality
                if modality in frame_data:
                    result = await self._process_modality_async(modality, frame_data[modality])
                    results[modality] = result
                    self.last_process_times[modality] = current_time

        # Apply temporal smoothing
        results = self._apply_temporal_smoothing(results)

        return results

    async def _process_modality_async(self, modality: str, data: Any) -> Any:
        """Process a single modality asynchronously"""
        loop = asyncio.get_event_loop()

        if modality == 'vision':
            return await loop.run_in_executor(self.executor, self._process_vision, data)
        elif modality == 'language':
            return await self._process_language_async(data)
        elif modality == 'gesture':
            return await loop.run_in_executor(self.executor, self._process_gesture, data)

        return data

    def _process_vision(self, image_data: np.ndarray) -> Dict:
        """Process visual data (runs in thread pool)"""
        # Perform heavy computer vision processing
        # This is where you'd call your vision models
        return {
            'objects': self._detect_objects(image_data),
            'scene_description': self._describe_scene(image_data),
            'timestamp': time.time()
        }

    async def _process_language_async(self, text: str) -> Dict:
        """Process language data asynchronously"""
        # This could call an LLM API
        # For now, simulate with a simple NLP task
        return {
            'intent': self._classify_intent(text),
            'entities': self._extract_entities(text),
            'sentiment': self._analyze_sentiment(text),
            'timestamp': time.time()
        }

    def _process_gesture(self, image_data: np.ndarray) -> Dict:
        """Process gesture data (runs in thread pool)"""
        # Perform gesture recognition
        return {
            'gesture': self._recognize_gesture(image_data),
            'confidence': 0.8,  # Simulated confidence
            'timestamp': time.time()
        }

    def _apply_temporal_smoothing(self, results: Dict) -> Dict:
        """Apply temporal smoothing to reduce jitter"""
        for modality, result in results.items():
            if modality == 'gesture':
                # Add to temporal buffer
                self.temporal_buffers['gestures'].append(result)

                # Apply smoothing if we have enough samples
                if len(self.temporal_buffers['gestures']) >= 3:
                    # Simple majority vote for gesture recognition
                    gestures = [g['gesture'] for g in self.temporal_buffers['gestures']]
                    most_common = max(set(gestures), key=gestures.count)
                    confidence = sum(1 for g in gestures if g == most_common) / len(gestures)

                    result['gesture'] = most_common
                    result['confidence'] = confidence

        return results

    def _detect_objects(self, image: np.ndarray) -> List[str]:
        """Detect objects in image (simplified)"""
        # In practice, this would use a trained object detection model
        return ['person', 'chair', 'table']  # Simulated results

    def _describe_scene(self, image: np.ndarray) -> str:
        """Describe scene in image (simplified)"""
        return "A typical indoor scene with furniture and people"

    def _classify_intent(self, text: str) -> str:
        """Classify intent of text (simplified)"""
        # Simple keyword-based classification
        text_lower = text.lower()
        if any(word in text_lower for word in ['navigate', 'go', 'move', 'walk']):
            return 'navigation'
        elif any(word in text_lower for word in ['grasp', 'pick', 'take', 'hold']):
            return 'manipulation'
        else:
            return 'conversation'

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text (simplified)"""
        # Simple entity extraction
        words = text.split()
        entities = [word for word in words if word.istitle() or word.lower() in ['person', 'object', 'location']]
        return entities

    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of text (simplified)"""
        positive_words = ['good', 'great', 'excellent', 'please', 'thank', 'hello']
        negative_words = ['bad', 'terrible', 'stop', 'no', 'not']

        pos_count = sum(1 for word in text.lower().split() if word in positive_words)
        neg_count = sum(1 for word in text.lower().split() if word in negative_words)

        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'

    def _recognize_gesture(self, image: np.ndarray) -> str:
        """Recognize gesture in image (simplified)"""
        # Simulated gesture recognition
        import random
        gestures = ['wave', 'point', 'grasp', 'stop']
        return random.choice(gestures)
```

### Resource Management
Implement resource management for multimodal systems:

```python
class ResourceManager:
    def __init__(self):
        self.gpu_memory_limit = 0.8  # Use up to 80% of GPU memory
        self.cpu_usage_limit = 0.8   # Use up to 80% of CPU
        self.active_processes = {}

        # Priority levels for different modalities
        self.modality_priorities = {
            'safety': 10,  # Highest priority
            'navigation': 8,
            'manipulation': 7,
            'gesture': 5,
            'vision': 4,
            'language': 3,
            'communication': 2,
            'idle': 1  # Lowest priority
        }

    def check_resource_availability(self, required_resources: Dict) -> bool:
        """Check if required resources are available"""
        import psutil
        import GPUtil

        # Check CPU usage
        cpu_percent = psutil.cpu_percent()
        if cpu_percent > self.cpu_usage_limit * 100:
            return False

        # Check memory usage
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 90:  # 90% memory usage threshold
            return False

        # Check GPU if available
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_load = gpus[0].load
            gpu_memory_used = gpus[0].memoryUtil
            if gpu_load > self.gpu_memory_limit or gpu_memory_used > self.gpu_memory_limit:
                return False

        return True

    def prioritize_processing(self, modality: str, priority_override: int = None) -> bool:
        """Determine if processing should proceed based on priority"""
        priority = priority_override or self.modality_priorities.get(modality, 1)

        # Check current system load
        import psutil
        system_load = psutil.cpu_percent()

        # High-priority tasks can proceed even under load
        if priority >= 8:
            return True

        # Medium-priority tasks proceed if load is reasonable
        if priority >= 5 and system_load < 70:
            return True

        # Low-priority tasks only proceed if system is lightly loaded
        if priority >= 2 and system_load < 50:
            return True

        return False

    def manage_processing_queue(self, tasks: List[Dict]) -> List[Dict]:
        """Manage processing queue based on priorities and resources"""
        # Sort tasks by priority
        sorted_tasks = sorted(tasks, key=lambda x: self.modality_priorities.get(x['modality'], 1), reverse=True)

        # Process tasks based on available resources
        processed_tasks = []
        for task in sorted_tasks:
            if self.check_resource_availability(task.get('resources', {})):
                if self.prioritize_processing(task['modality']):
                    processed_tasks.append(task)

        return processed_tasks
```

## Troubleshooting Common Issues

### Synchronization Problems
Handle timing issues between modalities:

```python
class SynchronizationManager:
    def __init__(self, max_delay_tolerance: float = 0.5):
        self.max_delay_tolerance = max_delay_tolerance
        self.modality_timestamps = {}
        self.synchronization_windows = {}

    def update_modality_timestamp(self, modality: str, timestamp: float):
        """Update timestamp for a modality"""
        self.modality_timestamps[modality] = timestamp

    def check_synchronization(self, modalities: List[str]) -> Dict[str, bool]:
        """Check if modalities are synchronized"""
        if len(modalities) < 2:
            return {mod: True for mod in modalities}

        # Find the most recent timestamp
        recent_time = max(self.modality_timestamps.get(mod, 0) for mod in modalities)

        # Check if all modalities are within tolerance
        sync_status = {}
        for mod in modalities:
            mod_time = self.modality_timestamps.get(mod, 0)
            delay = abs(recent_time - mod_time)
            sync_status[mod] = delay <= self.max_delay_tolerance

        return sync_status

    def get_synchronized_data(self, data_buffer: Dict, modalities: List[str]) -> Dict:
        """Get data from modalities that are synchronized"""
        sync_status = self.check_synchronization(modalities)

        synchronized_data = {}
        for mod in modalities:
            if sync_status.get(mod, False) and mod in data_buffer:
                synchronized_data[mod] = data_buffer[mod]

        return synchronized_data
```

### Handling Ambiguous Inputs
Implement strategies for dealing with ambiguous multimodal inputs:

```python
class AmbiguityResolver:
    def __init__(self):
        self.confidence_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }

    def resolve_ambiguity(self, multimodal_inputs: Dict) -> Dict:
        """Resolve ambiguities in multimodal inputs"""
        resolved_outputs = {}

        # Check for conflicting information between modalities
        conflicts = self._detect_conflicts(multimodal_inputs)

        if conflicts:
            # Use context and confidence to resolve conflicts
            resolved_outputs = self._resolve_conflicts(multimodal_inputs, conflicts)
        else:
            # No conflicts, use all inputs
            resolved_outputs = multimodal_inputs

        return resolved_outputs

    def _detect_conflicts(self, inputs: Dict) -> List[Dict]:
        """Detect conflicts between modalities"""
        conflicts = []

        # Example: Check if vision and language disagree on navigation target
        if 'vision' in inputs and 'language' in inputs:
            vision_target = inputs['vision'].get('navigation_target')
            language_target = inputs['language'].get('navigation_target')

            if vision_target and language_target and vision_target != language_target:
                conflicts.append({
                    'type': 'navigation_target_conflict',
                    'modalities': ['vision', 'language'],
                    'values': [vision_target, language_target]
                })

        # Add more conflict detection logic as needed

        return conflicts

    def _resolve_conflicts(self, inputs: Dict, conflicts: List[Dict]) -> Dict:
        """Resolve detected conflicts"""
        resolved = inputs.copy()

        for conflict in conflicts:
            if conflict['type'] == 'navigation_target_conflict':
                # Use weighted average based on confidence
                vision_conf = inputs['vision'].get('confidence', 0.5)
                language_conf = inputs['language'].get('confidence', 0.5)

                if vision_conf > language_conf:
                    resolved['navigation_target'] = inputs['vision']['navigation_target']
                else:
                    resolved['navigation_target'] = inputs['language']['navigation_target']

        return resolved

    def request_clarification(self, ambiguous_input: Dict) -> str:
        """Generate request for clarification of ambiguous input"""
        # Analyze the ambiguous input to determine what clarification is needed
        if 'action' in ambiguous_input and ambiguous_input['action'] == 'multiple_targets':
            return "I see multiple possible targets. Could you please specify which one you mean?"
        elif 'gesture' in ambiguous_input:
            return "I'm not sure what you mean by that gesture. Could you please clarify?"
        else:
            return "I didn't quite understand. Could you please repeat or clarify?"
```

## Best Practices

### 1. Modular Design
Design your multimodal system with clear separation of concerns:
- Separate processing pipelines for each modality
- Clear interfaces between components
- Independent testing capabilities for each module

### 2. Graceful Degradation
Ensure your system works even when some modalities fail:
- Fallback mechanisms for each modality
- Reduced functionality rather than complete failure
- Clear indication of which modalities are active

### 3. Context Awareness
Always consider the context when processing multimodal inputs:
- Task-based attention allocation
- Environmental factor consideration
- User intent and history

### 4. Performance Monitoring
Monitor and optimize performance continuously:
- Track processing times for each modality
- Monitor resource usage
- Identify bottlenecks and optimize

### 5. Safety First
Prioritize safety in all multimodal interactions:
- Safety checks before action execution
- Human-in-the-loop for critical decisions
- Clear error handling and recovery

### 6. User Experience
Design for natural and intuitive interactions:
- Consistent feedback across modalities
- Appropriate response times
- Clear communication of system state

## Exercises

1. Implement a vision-language model integration using CLIP or similar
2. Create a gesture recognition system using MediaPipe or OpenPose
3. Build a context-aware interaction system that maintains conversation history
4. Implement attention mechanisms for multimodal fusion
5. Create a ROS 2 node that integrates multiple sensory inputs
6. Add temporal consistency and smoothing to your multimodal system
7. Implement ambiguity resolution strategies for conflicting inputs
8. Design a resource management system for efficient multimodal processing
9. Build a multimodal input synchronization system to handle timing differences
10. Create a confidence-based fusion system that weights modalities by reliability
11. Implement a multimodal dialogue system that can handle back-and-forth interactions
12. Design and implement a system for handling missing or failed modalities gracefully

## Next Steps

After completing this chapter, you'll have a comprehensive understanding of multimodal interaction systems. Proceed to learn about implementing natural language to ROS 2 action translation to connect your multimodal understanding with robot action execution.