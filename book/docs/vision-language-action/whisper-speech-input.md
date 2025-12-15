# OpenAI Whisper Integration for Voice Command Processing

## Overview
This chapter covers the integration of OpenAI Whisper, a state-of-the-art speech recognition model, for processing voice commands in humanoid robots. Whisper enables accurate speech-to-text conversion that can be used as input for cognitive planning and action execution systems.

## Learning Objectives
- Understand OpenAI Whisper's architecture and capabilities
- Install and configure Whisper for real-time speech recognition
- Integrate Whisper with ROS 2 for voice command processing
- Optimize Whisper performance for embedded systems
- Handle various audio input formats and quality levels
- Implement error handling and confidence scoring

## Prerequisites
- Basic understanding of Python programming
- Knowledge of ROS 2 concepts and message types
- Familiarity with audio processing concepts
- Completed ROS 2 fundamentals and cognitive planning chapters

## Table of Contents
1. [Introduction to OpenAI Whisper](#introduction-to-openai-whisper)
2. [Installation and Setup](#installation-and-setup)
3. [Whisper Model Variants](#whisper-model-variants)
4. [Real-time Audio Processing](#real-time-audio-processing)
5. [ROS 2 Integration](#ros-2-integration)
6. [Performance Optimization](#performance-optimization)
7. [Error Handling and Confidence Scoring](#error-handling-and-confidence-scoring)
8. [Troubleshooting Common Issues](#troubleshooting-common-issues)
9. [Best Practices](#best-practices)

## Introduction to OpenAI Whisper

### What is OpenAI Whisper?
OpenAI Whisper is a robust speech recognition model trained on 680,000 hours of multilingual and multitask supervised data. It demonstrates strong performance across various domains, including low-resource languages, and shows resilience to accents, background noise, and technical language.

### Key Features
- **Multilingual Support**: Supports 99 languages
- **Robustness**: Performs well on diverse audio conditions
- **Timestamps**: Provides word-level and segment-level timestamps
- **Speaker Diarization**: Can identify different speakers (with additional tools)
- **Punctuation and Capitalization**: Automatically adds punctuation and capitalization

### Use Cases for Humanoid Robots
- Voice command interpretation
- Natural language interaction
- Accessibility features
- Multilingual communication
- Context-aware responses

## Installation and Setup

### System Requirements
- Python 3.8 or higher
- At least 6GB of RAM for large models
- CUDA-compatible GPU (recommended for real-time processing)
- Audio input device (microphone)

### Installation Steps
```bash
# Create a virtual environment
python -m venv whisper_env
source whisper_env/bin/activate  # On Windows: whisper_env\Scripts\activate

# Install Whisper
pip install openai-whisper

# Install additional dependencies for audio processing
pip install pyaudio soundfile

# For GPU acceleration (optional but recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Installing FFmpeg (Required for Audio Processing)
Whisper requires FFmpeg for audio format conversion:

**On Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**On macOS:**
```bash
brew install ffmpeg
```

**On Windows:**
Download from https://ffmpeg.org/download.html and add to PATH

## Whisper Model Variants

### Available Models
Whisper comes in five model sizes, each with different performance characteristics:

| Model | Parameters | English-only | Multilingual | Required VRAM | Relative Speed |
|-------|------------|--------------|--------------|---------------|----------------|
| tiny  | 39 M       | tiny.en      | tiny         | ~1 GB         | ~32x           |
| base  | 74 M       | base.en      | base         | ~1 GB         | ~16x           |
| small | 244 M      | small.en     | small        | ~2 GB         | ~6x            |
| medium| 769 M      | medium.en    | medium       | ~5 GB         | ~2x            |
| large | 1550 M     | N/A          | large        | ~10 GB        | 1x             |

### Model Selection for Humanoid Robots
For humanoid robots, consider these factors:
- **tiny/base**: Good for embedded systems with limited resources
- **small**: Balance between performance and resource usage
- **medium/large**: Best accuracy but require more computational resources

```python
import whisper

# Load different model sizes
model_tiny = whisper.load_model("tiny")
model_base = whisper.load_model("base")
model_small = whisper.load_model("small")
model_medium = whisper.load_model("medium")
model_large = whisper.load_model("large")
```

## Real-time Audio Processing

### Audio Capture with PyAudio
For real-time voice command processing, we need to capture audio from a microphone:

```python
import pyaudio
import numpy as np
import wave
import threading
import queue
import whisper
import torch

class AudioCapture:
    def __init__(self, chunk_size=1024, format=pyaudio.paInt16, channels=1, rate=16000):
        self.chunk_size = chunk_size
        self.format = format
        self.channels = channels
        self.rate = rate
        self.audio_queue = queue.Queue()

        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.recording = False

    def start_recording(self):
        """Start audio recording"""
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        self.recording = True
        self.record_thread = threading.Thread(target=self._record_audio)
        self.record_thread.start()

    def stop_recording(self):
        """Stop audio recording"""
        self.recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def _record_audio(self):
        """Internal method to record audio in a separate thread"""
        while self.recording:
            data = self.stream.read(self.chunk_size)
            audio_data = np.frombuffer(data, dtype=np.int16)
            self.audio_queue.put(audio_data)

    def get_audio_data(self):
        """Get accumulated audio data from the queue"""
        audio_frames = []
        while not self.audio_queue.empty():
            audio_frames.append(self.audio_queue.get())

        if audio_frames:
            return np.concatenate(audio_frames)
        return np.array([])
```

### Audio Preprocessing for Whisper
Whisper expects audio in a specific format. Here's how to prepare audio data:

```python
import librosa
import soundfile as sf
import tempfile
import os

def preprocess_audio_for_whisper(audio_data, sample_rate=16000):
    """
    Preprocess audio data for Whisper model
    """
    # Ensure audio is in the right format
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize from int16

    # Resample if needed
    if sample_rate != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)

    return audio_data

def save_audio_to_temp_file(audio_data, sample_rate=16000):
    """
    Save audio data to a temporary WAV file for Whisper processing
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        sf.write(temp_file.name, audio_data, sample_rate)
        return temp_file.name
```

### Real-time Speech Recognition Pipeline
Here's a complete example of a real-time speech recognition pipeline:

```python
import whisper
import numpy as np
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, Callable

@dataclass
class SpeechRecognitionResult:
    text: str
    confidence: float
    language: str
    timestamp: float

class RealTimeWhisper:
    def __init__(self, model_size="small", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = whisper.load_model(model_size).to(device)
        self.device = device
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.running = False
        self.recognition_thread = None

    def start_recognition(self):
        """Start the real-time recognition thread"""
        self.running = True
        self.recognition_thread = threading.Thread(target=self._process_audio)
        self.recognition_thread.start()

    def stop_recognition(self):
        """Stop the recognition process"""
        self.running = False
        if self.recognition_thread:
            self.recognition_thread.join()

    def add_audio_chunk(self, audio_chunk):
        """Add an audio chunk to be processed"""
        self.audio_queue.put(audio_chunk)

    def get_result(self) -> Optional[SpeechRecognitionResult]:
        """Get the next recognition result"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def _process_audio(self):
        """Internal method to process audio chunks"""
        accumulated_audio = np.array([])

        while self.running:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_queue.get(timeout=0.1)

                # Accumulate audio
                accumulated_audio = np.concatenate([accumulated_audio, audio_chunk])

                # Process if we have enough audio (1 second worth at 16kHz)
                if len(accumulated_audio) >= 16000:
                    # Process the accumulated audio
                    result = self._recognize_speech(accumulated_audio)
                    if result:
                        self.result_queue.put(result)

                    # Keep the last 0.5 seconds to maintain continuity
                    accumulated_audio = accumulated_audio[-8000:]

            except queue.Empty:
                continue

    def _recognize_speech(self, audio_data):
        """Recognize speech in the given audio data"""
        try:
            # Convert to float32 and normalize
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / 32768.0

            # Run Whisper
            result = self.model.transcribe(audio_data, language="en", without_timestamps=True)

            # Calculate confidence based on compression ratio and other factors
            confidence = self._calculate_confidence(result)

            return SpeechRecognitionResult(
                text=result["text"],
                confidence=confidence,
                language="en",  # This could be dynamic
                timestamp=time.time()
            )
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return None

    def _calculate_confidence(self, result):
        """Calculate confidence score based on Whisper's internal metrics"""
        # Whisper provides some metrics that can be used for confidence
        # This is a simplified version - in practice, you might want to use more sophisticated methods
        compression_ratio = result.get("compression_ratio", 1.0)

        # Lower compression ratios might indicate less reliable transcriptions
        if compression_ratio > 2.5:
            return 0.3  # Low confidence
        elif compression_ratio > 1.8:
            return 0.7  # Medium confidence
        else:
            return 0.9  # High confidence
```

## ROS 2 Integration

### Creating a Whisper ROS 2 Node
To integrate Whisper with ROS 2, we need to create a node that can process audio and publish recognized text:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String, Float32
from sensor_msgs.msg import AudioData
from vision_language_action_msgs.msg import SpeechRecognitionResult  # Custom message
import whisper
import numpy as np
import torch

class WhisperNode(Node):
    def __init__(self):
        super().__init__('whisper_node')

        # Initialize Whisper model
        self.model = whisper.load_model("small")
        self.get_logger().info("Whisper model loaded")

        # Create publishers
        self.text_pub = self.create_publisher(String, 'recognized_text', 10)
        self.result_pub = self.create_publisher(SpeechRecognitionResult, 'speech_result', 10)
        self.confidence_pub = self.create_publisher(Float32, 'recognition_confidence', 10)

        # Create subscriber for audio data
        self.audio_sub = self.create_subscription(
            AudioData,
            'audio_input',
            self.audio_callback,
            QoSProfile(depth=10)
        )

        self.get_logger().info("Whisper node initialized")

    def audio_callback(self, msg):
        """Callback for audio data"""
        try:
            # Convert audio data to numpy array
            audio_array = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0

            # Perform speech recognition
            result = self.model.transcribe(audio_array, language="en")

            # Publish recognized text
            text_msg = String()
            text_msg.data = result["text"]
            self.text_pub.publish(text_msg)

            # Calculate and publish confidence
            confidence = self._calculate_confidence(result)
            confidence_msg = Float32()
            confidence_msg.data = confidence
            self.confidence_pub.publish(confidence_msg)

            # Publish detailed result
            result_msg = SpeechRecognitionResult()
            result_msg.text = result["text"]
            result_msg.confidence = confidence
            result_msg.language = "en"
            result_msg.timestamp = self.get_clock().now().to_msg()
            self.result_pub.publish(result_msg)

            self.get_logger().info(f"Recognized: {result['text']} (confidence: {confidence:.2f})")

        except Exception as e:
            self.get_logger().error(f"Error processing audio: {e}")

    def _calculate_confidence(self, result):
        """Calculate confidence score"""
        compression_ratio = result.get("compression_ratio", 1.0)

        if compression_ratio > 2.5:
            return 0.3
        elif compression_ratio > 1.8:
            return 0.7
        else:
            return 0.9

def main(args=None):
    rclpy.init(args=args)
    node = WhisperNode()

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

### Audio Input Node
We also need a node to capture audio and publish it to the audio_input topic:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import AudioData
import pyaudio
import numpy as np
import threading

class AudioInputNode(Node):
    def __init__(self):
        super().__init__('audio_input_node')

        # Audio configuration
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000

        # Create publisher
        self.audio_pub = self.create_publisher(AudioData, 'audio_input', 10)

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.recording = False

        # Start recording in a separate thread
        self.record_thread = threading.Thread(target=self._record_audio)
        self.recording = True
        self.record_thread.start()

        self.get_logger().info("Audio input node started")

    def _record_audio(self):
        """Record audio in a separate thread"""
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        while self.recording:
            try:
                data = self.stream.read(self.chunk_size)
                audio_msg = AudioData()
                audio_msg.data = data
                self.audio_pub.publish(audio_msg)
            except Exception as e:
                self.get_logger().error(f"Error recording audio: {e}")
                break

    def destroy_node(self):
        self.recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = AudioInputNode()

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

### GPU Acceleration
For better performance on humanoid robots with GPU capabilities:

```python
import whisper
import torch

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("small").to(device)

# Process audio with GPU acceleration
result = model.transcribe(audio_data, language="en")
```

### Model Quantization
For resource-constrained systems, consider using quantized models:

```python
# Load quantized model for faster inference on CPU
model = whisper.load_model("small", device="cpu", in_memory=True)
# Note: Whisper doesn't have explicit quantization, but you can use smaller models
```

### Audio Buffer Management
Efficient audio buffer management for real-time processing:

```python
import collections
import numpy as np

class AudioBuffer:
    def __init__(self, max_duration=5.0, sample_rate=16000):
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = collections.deque(maxlen=self.max_samples)

    def add_audio(self, audio_chunk):
        """Add audio chunk to buffer"""
        for sample in audio_chunk:
            self.buffer.append(sample)

    def get_recent_audio(self, duration=1.0):
        """Get recent audio of specified duration"""
        samples_needed = int(duration * self.sample_rate)
        start_idx = max(0, len(self.buffer) - samples_needed)
        return np.array(list(self.buffer)[start_idx:])

    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
```

## Error Handling and Confidence Scoring

### Confidence Thresholds
Implement confidence thresholds to filter unreliable transcriptions:

```python
CONFIDENCE_THRESHOLD_HIGH = 0.8
CONFIDENCE_THRESHOLD_MEDIUM = 0.5

def is_reliable_transcription(result, threshold=CONFIDENCE_THRESHOLD_MEDIUM):
    """Check if transcription is reliable based on confidence"""
    confidence = _calculate_confidence(result)
    return confidence >= threshold

def handle_recognition_result(result):
    """Handle recognition result based on confidence"""
    confidence = _calculate_confidence(result)
    text = result["text"].strip()

    if not text:
        return {"action": "ignore", "reason": "empty_transcription"}

    if confidence >= CONFIDENCE_THRESHOLD_HIGH:
        return {"action": "process", "text": text, "confidence": "high"}
    elif confidence >= CONFIDENCE_THRESHOLD_MEDIUM:
        return {"action": "process_with_caution", "text": text, "confidence": "medium"}
    else:
        return {"action": "ignore", "reason": "low_confidence", "confidence": confidence}
```

### Error Recovery Strategies
Handle common errors in speech recognition:

```python
def robust_transcribe(model, audio_data, max_retries=3):
    """Transcribe audio with error recovery"""
    for attempt in range(max_retries):
        try:
            # Preprocess audio
            if len(audio_data) == 0:
                return {"text": "", "error": "empty_audio"}

            # Ensure minimum audio length
            if len(audio_data) < 16000:  # 1 second at 16kHz
                # Pad with zeros if too short
                padded = np.zeros(16000, dtype=audio_data.dtype)
                padded[:len(audio_data)] = audio_data
                audio_data = padded

            result = model.transcribe(audio_data, language="en", without_timestamps=True)
            return result

        except Exception as e:
            print(f"Transcription attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return {"text": "", "error": str(e)}

            # Brief pause before retry
            time.sleep(0.1)
```

## Troubleshooting Common Issues

### Audio Format Issues
Ensure audio is in the correct format for Whisper:

```python
def validate_audio_format(audio_data, sample_rate):
    """Validate audio format for Whisper compatibility"""
    # Check sample rate
    if sample_rate != 16000:
        print(f"Warning: Sample rate is {sample_rate}, Whisper expects 16000Hz")

    # Check data type
    if audio_data.dtype != np.float32:
        print(f"Warning: Audio data type is {audio_data.dtype}, converting to float32")
        audio_data = audio_data.astype(np.float32)

    # Check for silent audio
    if np.all(audio_data == 0):
        print("Warning: Audio appears to be silent")

    return audio_data
```

### Memory Issues
Handle memory constraints on embedded systems:

```python
import gc

def process_audio_with_memory_management(model, audio_data):
    """Process audio with explicit memory management"""
    try:
        result = model.transcribe(audio_data, language="en")
        return result
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Try with smaller model or shorter audio
            print("Out of memory, consider using smaller model or shorter audio segments")
            return {"text": "", "error": "out_of_memory"}
        else:
            raise e
```

## Best Practices

1. **Model Selection**: Choose the appropriate model size based on your hardware capabilities
2. **Audio Quality**: Use high-quality microphones and audio preprocessing
3. **Confidence Thresholds**: Implement confidence-based filtering for reliable results
4. **Error Handling**: Implement robust error handling and recovery mechanisms
5. **Resource Management**: Monitor and manage memory and computational resources
6. **Testing**: Test with various audio conditions (noise, accents, etc.)
7. **Latency**: Balance accuracy with real-time processing requirements

## Exercises

1. Install OpenAI Whisper and test it with sample audio files
2. Create a ROS 2 node that captures audio from a microphone and publishes it
3. Implement a speech recognition node that uses Whisper to transcribe audio
4. Add confidence scoring and threshold filtering to your recognition system
5. Optimize your Whisper implementation for performance on your target hardware
6. Test your system with different audio conditions and noise levels
7. Implement a voice activity detection system to reduce unnecessary processing
8. Create a multi-language speech recognition system using Whisper's multilingual capabilities
9. Integrate audio preprocessing (noise reduction, normalization) with Whisper
10. Build a real-time streaming audio recognition system

## Next Steps

After completing this chapter, proceed to learn about LLM cognitive planning to understand how to process the recognized speech into actionable commands for your humanoid robot.