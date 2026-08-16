# Android Mobile Prototype Plan

## Goal

Build a minimal Android-focused extension of the scam detection system that can analyze a live phone conversation without directly accessing cellular call audio.

The practical user flow is:

```text
User answers a call
  -> app asks whether to analyze scam risk
  -> user opts in
  -> app asks user to put the call on speaker
  -> app records microphone audio
  -> local ASR + local classifier inference
  -> app displays scam-risk warning
```

This avoids relying on restricted access to internal call audio and keeps the privacy story clean.

## Scope

Target platform:

```text
Android
```

Initial demo mode:

```text
Manual start from app
```

Optional later mode:

```text
Call-state aware notification when a call is active
```

The first prototype should not attempt to capture telephony audio directly.

## Proposed Runtime Stack

```text
ASR:
  whisper.cpp tiny Q4_K or Q8_0
  runs on CPU

Classifier:
  ModernBERT transcript classifier exported to ONNX
  calibrated INT8 PTQ using ptq_calibration.csv
  runs through ONNX Runtime Mobile + NNAPI

Target acceleration:
  Google Tensor / Android NNAPI path

Fallback:
  ONNX Runtime CPU execution provider
```

GGUF remains the main CPU deployment path for the current Hugging Face/local demo. ONNX INT8 PTQ becomes the mobile/NPU-oriented path.

## Streaming Architecture

Use overlapping workers so ASR and classification do not block each other.

```text
AudioRecord thread
  -> captures 5-10 second PCM chunks
  -> writes chunks to audio_queue

CPU ASR worker
  -> consumes audio_queue
  -> runs whisper.cpp
  -> emits transcript chunks
  -> appends to rolling transcript buffer

NPU classifier worker
  -> consumes rolling transcript windows
  -> runs ONNX INT8 ModernBERT classifier through NNAPI
  -> emits scam probability

UI thread
  -> displays transcript, risk score, warning state
```

Overlapped execution:

```text
CPU transcribes audio chunk N+1
while
NPU classifies transcript window from chunk N
```

## Inference Strategy

Do not classify every tiny transcript fragment in isolation. Scam intent may emerge across multiple turns.

Recommended approach:

```text
audio_chunk_seconds = 10
rolling_transcript_window = last 60-90 seconds
classification_interval = every 10-20 seconds
```

Classifier input:

```text
latest rolling transcript window
```

Output:

```text
scam_probability
label: Scam / Legitimate
```

Risk smoothing:

```text
smoothed_risk = 0.7 * previous_risk + 0.3 * current_probability
```

Trigger warning only when risk is stable:

```text
smoothed_risk > 0.8 for 2 consecutive windows
```

This avoids warning the user from a single noisy ASR chunk.

## Permissions

Required:

```text
RECORD_AUDIO
FOREGROUND_SERVICE
POST_NOTIFICATIONS
```

Optional:

```text
READ_PHONE_STATE
```

`READ_PHONE_STATE` is only needed for a later version that detects active calls and shows a notification. The minimal demo can avoid this and use manual start.

## User Experience

Minimal demo flow:

```text
1. User opens app.
2. User taps "Start Scam Check".
3. App shows: "Put your call on speaker. Audio stays on device."
4. App starts foreground microphone recording.
5. App shows rolling transcript.
6. App shows scam risk score.
7. If risk remains high, app displays warning.
```

Suggested UI states:

```text
Idle
Listening
Transcribing
Analyzing
Likely Legitimate
High Scam Risk
```

## Privacy Position

The mobile extension should be positioned as privacy-first:

```text
- user explicitly opts in per call
- app uses microphone only after permission
- user must place call on speaker
- audio and transcripts stay on device
- chunks can be discarded after inference
- no cloud API is required for inference
```

This is stronger than a server-based call monitoring system because sensitive phone-call content does not need to leave the device.

## Role of PTQ

Two optimization paths serve different deployment goals:

```text
GGUF weight-only PTQ:
  primary CPU/local deployment path
  used for Hugging Face Spaces and local CPU inference

ONNX calibrated INT8 PTQ:
  mobile/NPU-oriented path
  uses data/processed/ptq_calibration.csv
  intended for ONNX Runtime Mobile + NNAPI
```

The ONNX INT8 path does not need to be the final HF Spaces deployment format. Its value is showing a realistic bridge toward Android NPU inference.

## Implementation Phases

### V0: Manual Android Demo

```text
Open app
  -> tap Start
  -> microphone records chunks
  -> CPU whisper.cpp transcribes
  -> ONNX classifier predicts
  -> UI shows risk
```

No call-state detection.

### V1: Call-Aware Prompt

```text
Detect active phone call
  -> show notification: "Analyze this call?"
  -> user taps yes
  -> app starts microphone-based analysis
```

Requires additional Android permissions and careful UX.

### V2: Real-Time Risk Updates

```text
streaming transcript
rolling risk score
warning threshold
optional explanation snippets
```

## Main Risks

```text
ASR latency:
  Whisper is the heaviest component. Use tiny Q4/Q8 and chunked processing.

Audio quality:
  Speakerphone + microphone can introduce noise. Use short chunks and risk smoothing.

NPU compatibility:
  NNAPI support varies by device. Keep CPU fallback.

Tokenization:
  ONNX classifier still needs mobile-side tokenization before inference.

False alarms:
  Use rolling context and consecutive-window thresholding.
```

## How To Present In Final Work

This should be presented as a focused future/mobile extension:

```text
Android prototype direction:
CPU whisper.cpp handles streaming ASR.
ONNX INT8 ModernBERT classifier runs through NNAPI on Google Tensor-style devices.
The two workers overlap so the CPU can continue transcribing while the NPU classifies the latest transcript window.
```

Do not present it as completed unless implemented. Present it as the natural next step enabled by the current GGUF and ONNX/PTQ optimization work.
