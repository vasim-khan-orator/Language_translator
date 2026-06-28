# Fix Plan and Architecture Evolution — Language Translator

This document outlines the remediation roadmap and subsequent architectural evolution of the real-time speech translation pipeline. All emojis have been removed for clean ASCII/UTF-8 rendering across environments.

---

## Architectural Evolution Note (Final Update)

Following the initial completion of Phases 1 through 5, real-world testing revealed that a rolling-window architecture (3-second windows emitted every 250ms) inherently caused non-deterministic word splitting and hypothesis diffing conflicts.

To resolve this permanently, the project underwent a complete redesign:
1. **Migration to VAD-Gated Audio Accumulation**: Replaced fixed 3-second circular ring buffers with raw 20ms chunk streaming. A state machine (`IDLE` -> `ACCUMULATING`) records the exact duration of spoken sentences.
2. **Removal of Hypothesis Diffing**: Deleted `hypothesis_manager.py` and single-word translations. Live previews now translate accumulated phrases directly.
3. **GPU Acceleration**: Added NVIDIA CUDA acceleration for faster-whisper and IndicTrans2, reducing latency from ~1.5s on CPU to ~150ms on GPU.

---

## Phase Summary (Historical Fix Roadmap)

| Phase | Focus | Issues | Status |
|-------|-------|--------|--------|
| 1 | Critical Bugs | #1.1, #1.2, #1.3 | COMPLETED |
| 2 | Performance | #2.1, #2.2, #2.3 | COMPLETED |
| 3 | Threading | #3.1, #3.2, #3.3 | COMPLETED |
| 4 | Architecture | #4.1, #4.2, #4.3 | COMPLETED |
| 5 | Polish | #5.1 to #5.8 | COMPLETED |

---

## Verification Checklist

- [x] Speak -> pause 1.5s -> sentence finalizes correctly via VAD silence detection
- [x] No crashes under sustained multi-minute conversations (model locks and GPU management)
- [x] Memory bounded (audio queue maxsize=500 for 10s streaming buffer)
- [x] Ctrl-C exits cleanly and releases microphone resources
- [x] Live preview shows readable, grammatically correct phrase translations
- [x] All dependencies install cleanly from `requirements.txt`
