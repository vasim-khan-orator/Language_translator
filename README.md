# Realtime Hindi to English Translator

A streaming, offline-first speech translation pipeline that captures live Hindi speech from a microphone, transcribes it, translates it to English, and renders rolling subtitles in real time. Designed with a robust Voice Activity Detection (VAD) audio accumulation architecture and hardware acceleration support.

---

## Architecture Overview

The system uses a VAD-gated audio accumulation state machine to eliminate common real-time speech recognition artifacts such as echoing, word jumbling, and sentence clipping.

```
[Microphone]
     |
     v
+-------------------------------------------------------+
| audio_listener.py                                     |
| Continuously streams raw 20ms audio chunks (16kHz)    |
+---------------------------+---------------------------+
                            | audio_queue
                            v
+-------------------------------------------------------+
| main.py : speech_recognition_worker                   |
| VAD State Machine (IDLE -> ACCUMULATING)              |
| - Uses Silero VAD to detect speech start and pauses   |
| - Live Preview: Transcribes buffer every ~750ms       |
| - Finalization: Transcribes full utterance after 1.5s |
+---------------------------+---------------------------+
                            | text_queue (live & final)
                            v
+-------------------------------------------------------+
| main.py : word_processing_worker                      |
| - Live Preview: Translates phrases in real time       |
| - Final Output: Context-aware full translation        |
+---------------------------+---------------------------+
                            |
                            v
+-------------------------------------------------------+
| output_renderer.py                                    |
| In-place 7-line rolling terminal subtitle display     |
+-------------------------------------------------------+
```

### Two-Pass Translation Strategy

| Pass | Trigger | Method | Purpose |
|------|---------|--------|---------|
| **Live Preview** | While speaking (every ~750ms) | `translate_phrase()` on accumulated buffer | Immediate real-time feedback with correct Subject-Object-Verb (SOV) structure |
| **Final Output** | 1.5s silence pause or hard timeout | `reconstruct_sentence()` on complete utterance | Grammatically polished English translation |

---

## Features

- **VAD-Gated Accumulation**: Eliminates fixed rolling-window clipping by recording the exact duration of each spoken sentence from start to finish.
- **CUDA Acceleration**: Automatically detects and utilizes NVIDIA GPUs (PyTorch CUDA + faster-whisper float16) for low-latency inference, falling back cleanly to CPU when unavailable.
- **Phrase-Level Translation**: Translates complete phrases during live previews to maintain proper grammar across language structural differences (Hindi SOV to English SVO).
- **Rolling Terminal Subtitles**: Uses ANSI cursor controls to maintain a clean 7-line fixed subtitle display with live captions and finalized sentence history.

---

## Tech Stack

| Component | Technology | Description |
|-----------|-----------|-------------|
| Speech-to-Text | faster-whisper (`small` model) | Hindi speech recognition |
| Voice Activity Detection | Silero VAD | Speech activity gating and pause detection |
| Translation | IndicTrans2 200M | Neural Hindi to English translation |
| Audio Capture | PyAudio | 16 kHz mono microphone streaming |
| Runtime | Python 3.10+ | Multi-threaded queue architecture |

---

## Requirements

- Python 3.10 or later
- Microphone
- Approximately 2 GB RAM / VRAM for model loading
- NVIDIA GPU with CUDA support (Recommended for optimal speed)

---

## Installation

1. Clone the repository and navigate into the project directory:
```bash
git clone https://github.com/<your-username>/Language_translator.git
cd Language_translator
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

### PyAudio Installation Troubleshooting

If `pyaudio` fails to install on Windows:
```bash
pip install pipwin
pipwin install pyaudio
```

On Linux (Debian/Ubuntu):
```bash
sudo apt-get install portaudio19-dev python3-dev
pip install pyaudio
```

---

## Usage

1. Activate your virtual environment.
2. Run the main entry point:
```bash
python main.py
```

### Terminal Display Layout

When running, the application displays a fixed 7-line subtitle area:

```
==================================================
  >  [live Hindi words appear here as you speak]
     [live English translations appear here    ]
--------------------------------------------------
  मैं बाज़ार गया था  ->  I went to the market
  कल बारिश होगी      ->  It will rain tomorrow
==================================================
```

Press `Ctrl+C` to safely exit and stop audio recording.

---

## Project Structure

```
Language_translator/
|-- main.py                  # Orchestrates audio, STT, and processing threads
|-- audio_listener.py        # Raw 20ms microphone audio chunk capture
|-- vad_engine.py            # Voice Activity Detection wrapper (Silero)
|-- stt_engine.py            # Speech-to-Text wrapper (faster-whisper)
|-- translator_engine.py     # Neural translation engine (IndicTrans2)
|-- correction_engine.py     # Sentence reconstruction and finalization
|-- silence_manager.py       # Utterance boundary and pause timing
|-- token_buffer.py          # Semantic token buffer storage
|-- word_parser.py           # Text cleaning and preprocessing
|-- context_manager.py       # Conversation history tracking
|-- output_renderer.py       # ANSI terminal subtitle rendering
|-- requirements.txt         # Project dependencies
|-- FIX_PLAN.md              # Architecture history and remediation roadmap
`-- README.md                # Project documentation
```
