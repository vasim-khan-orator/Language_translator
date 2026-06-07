# Translator Project

A small offline-first translator pipeline combining voice activity detection (VAD), speech-to-text (STT), translation, and correction layers with a minimal UI.

- **Purpose**: Quickly experiment with streaming/transcription-based translation flows.
- **Key modules**: `stt_engine.py`, `vad_engine.py`, `translator_engine.py`, `correction_engine.py`, `output_renderer.py`.

**Requirements**
- Python 3.10+ (recommended)
- Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Quick Start**
- Launch the app:

```bash
source .venv/bin/activate
python main.py
```

**Useful files**
- **Entry point**: [main.py](main.py)
- **Services**: [services/app_launcher.py](services/app_launcher.py)
- **UI**: [ui/main_window.py](ui/main_window.py)

**Contributing**
- Open issues or PRs; keep changes small and focused.

**License**
- (Add your preferred license here)
