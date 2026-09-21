# Face-Control (脸控语音一体化无障碍控制系统)

<div align="center">

**Accessible Hands-Free Computer Control via Facial Pose, Expression & Offline Whisper Voice**  
*Control the mouse cursor, click buttons, and dictate text using standard webcams and microphones.*

[Key Capabilities](#key-capabilities) • [Cursor Control Mechanics](#cursor-tracking--interaction-mechanics) • [Voice & Whisper Engine](#voice-control--whisper-stt) • [Technical Architecture](#technical-architecture) • [Getting Started](#getting-started) • [Configuration Guide](#configuration-guide) • [Accessibility Innovations](#accessibility-design-principles)

</div>

---

## Overview

**Face-Control** is a hands-free accessibility system engineered for individuals with upper-limb mobility impairments, spinal injuries, or motor limitations. By converting facial pose geometry and vocal commands into standard system inputs, the platform enables complete, autonomous computer interaction using standard commodity hardware—requiring only an ordinary USB webcam and microphone without specialized eye-trackers or costly assistive equipment.

The application leverages **MediaPipe Face Mesh** to track 468 3D facial landmarks in real time, mapping nose-tip coordinates and head rotations to cursor trajectories. Mouth-aperture temporal analysis distinguishes between left and right mouse clicks, while local **OpenAI Whisper** models transcribe speech into text and process hands-free operational commands.

---

## Key Capabilities

- 🎯 **Subtle Head-Motion Tracking**: Maps nose-tip coordinates to screen pixel positions with customizable gain multipliers, allowing minimal head movements to span ultra-wide monitors without causing neck strain.
- 🪑 **One-Click Natural Posture Calibration**: Sets the user's current head position as the screen center with a single click, accommodating ergonomic sitting angles and non-standard postures.
- 🌊 **Adaptive Low-Pass Damping Filter**: Emphasizes cursor stability using a first-order low-pass filter ($\alpha = 0.001 - 0.99$) that eliminates tremor jitter while maintaining responsive tracking.
- 👄 **Duration-Based Mouth Click Triggering**:
  - *Brief mouth opening ($< 1.0\text{ s}$)*: Triggers a standard Left Click for selection and navigation.
  - *Sustained mouth opening ($\ge 2.0\text{ s}$)*: Triggers a Right Click for context menus.
  - Anti-chatter debounce intervals prevent accidental repetitive triggering.
- 🎙️ **Local Offline Whisper Dictation**: Integrates OpenAI Whisper (`tiny`, `base`, `small`, `medium`) for offline speech-to-text dictation across English and Chinese without cloud latency or subscription paywalls.
- 🗣️ **Spoken Execution Commands**:
  - Say *"Pause"* / *"Stop"* (or *"暂停"* / *"停止"*): Suspends facial tracking.
  - Say *"Resume"* / *"Start"* (or *"开始"* / *"继续"*): Re-engages facial tracking.
- 🎚️ **Input Device Routing & Noise Gating**: Enumerates all hardware audio input channels (USB mics, headset mics, webcam mics) and provides an RMS energy threshold filter to eliminate ambient noise before transcription.
- 🖥️ **Accessible PyQt5 Interface**: Features high-contrast camera monitoring, real-time landmark overlays, status badges, and an Always-On-Top viewing mode.

---

## Cursor Tracking & Interaction Mechanics

### 1. Nose-Tip Coordinate Mapping

The system tracks landmark `#4` (nose tip) across consecutive camera frames. An adjustable gain amplifier scales spatial displacements relative to the calibrated center point:

$$X_{\text{screen}} = \text{Screen Width} \times \left(0.5 + \text{Gain} \times \frac{x_{\text{nose}} - x_{\text{center}}}{\text{Frame Width}}\right)$$

$$Y_{\text{screen}} = \text{Screen Height} \times \left(0.5 + \text{Gain} \times \frac{y_{\text{nose}} - y_{\text{center}}}{\text{Frame Height}}\right)$$

### 2. Low-Pass Damping Filter

To balance cursor stability against responsiveness, filtered screen coordinates are updated per frame using damping factor $\alpha$:

$$\mathbf{P}_{\text{filtered}}(t) = \alpha \cdot \mathbf{P}_{\text{raw}}(t) + (1 - \alpha) \cdot \mathbf{P}_{\text{filtered}}(t - 1)$$

- Lower values ($\alpha \approx 0.05 - 0.2$): Maximize smoothness, ideal for users with motor tremors.
- Higher values ($\alpha \approx 0.5 - 0.9$): Maximize tracking immediacy for rapid navigation.

### 3. Vertical Mouth Aperture Analysis

Aperture is calculated from the Euclidean distance between upper lip landmark `#13` and lower lip landmark `#14`, normalized against facial height (distance between forehead landmark `#10` and chin landmark `#152`):

$$\text{Aperture Ratio} = \frac{\|\mathbf{p}_{13} - \mathbf{p}_{14}\|}{\|\mathbf{p}_{10} - \mathbf{p}_{152}\|}$$

When this ratio exceeds the configured trigger threshold, an internal timer initiates:
- Release within $< 1.0\text{ s}$ $\longrightarrow$ `pyautogui.click(button='left')`
- Sustained for $\ge 2.0\text{ s}$ $\longrightarrow$ `pyautogui.click(button='right')`

---

## Voice Control & Whisper STT

```text
Microphone Stream (sounddevice)
         │
         ▼
[RMS Energy Gate] ──► Discards silent buffers (< Threshold)
         │
         ▼
[Circular Audio Buffer]
         │
         ▼
[QThread Background Worker]
         │
         ├──► Whisper Local Model Inference
         │
         ├──► System Commands Check:
         │      ├── "Pause" / "Stop"   ──► Disable face cursor
         │      └── "Resume" / "Start" ──► Re-enable face cursor
         │
         └──► Dictation Routing:
                ├── UI Output Box display
                └── Keyboard typing / Clipboard paste at active cursor
```

---

## Technical Architecture

- **Primary Runtime**: Python 3.9+
- **Desktop Graphical Interface**: PyQt5
- **Computer Vision**: OpenCV (video capture & drawing), MediaPipe (468-point Face Mesh)
- **Mouse & Keyboard Synthesis**: `pyautogui`, `pyperclip`
- **Audio Capture**: `sounddevice`, `numpy`
- **Speech-to-Text Engine**: `openai-whisper`, `torch`
- **Concurrency**: Qt Event Loop with dedicated worker threads (`QThread`) isolating Whisper inference from real-time 30 FPS camera rendering

---

## Getting Started

### Prerequisites

- Windows 10/11, macOS, or Linux
- Python 3.9, 3.10, or 3.11
- Standard webcam and microphone

### Installation

```bash
# Clone the repository
git clone https://github.com/Karl-XZ/Face-Control.git
cd Face-Control

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install opencv-python mediapipe PyQt5 pyautogui pyperclip sounddevice numpy openai-whisper torch
```

### Launching the Application

```bash
python face_mouse.py
```

---

## Configuration Guide

| UI Parameter | Adjustment Method | Recommended Setting | Purpose |
| :--- | :--- | :--- | :--- |
| **Smoothing Alpha** | Slider + Text Box ($0.001 - 0.99$) | $0.15 - 0.30$ | Balances cursor smoothness against movement responsiveness. |
| **Cursor Gain** | Configuration Setting | $1.5 - 2.5$ | Amplifies head movement range to cover wide monitors. |
| **Center Pose** | Button (`Set Current Pose`) | Trigger at natural seating posture | Calibrates origin coordinates to eliminate neck fatigue. |
| **Audio Threshold** | Slider ($0.001 - 0.100$) | $0.015 - 0.030$ | Gating threshold preventing background noise from triggering STT. |
| **Microphone Channel** | Dropdown Selector | Dedicated USB Mic | Selects active hardware input device with immediate stream restart. |
| **Whisper Model** | Dropdown (`tiny` to `medium`) | `base` or `small` | Balances transcription accuracy against local GPU/CPU inference speed. |
| **Always on Top** | Checkbox Toggle | Enabled | Keeps camera view and status monitor visible over active application windows. |

---

## Accessibility Design Principles

1. **Zero Specialized Hardware Overhead**: Standard consumer webcams replace expensive proprietary eye trackers.
2. **Posture-Agnostic Operation**: The one-tap center calibration accommodates natural body positions, wheelchair recline angles, and asymmetric seating.
3. **Dual Confirmation Channels**: Clear visual feedback indicators (`Listening`, `Tracking Active`, `Paused`) ensure predictable interactions.
4. **Bilingual Accessibility**: Synchronized interface and language models support native English and Chinese workflows.
