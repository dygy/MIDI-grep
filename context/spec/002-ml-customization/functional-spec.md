# Functional Spec: ML Model Customization

**Feature ID:** 002-ml-customization
**Status:** Research Complete
**Priority:** High

---

## 1. Overview

Enable users to fine-tune the audio-to-MIDI transcription model for specific genres, instruments, or their own recordings. This improves accuracy for niche use cases where the stock Basic Pitch model underperforms.

---

## 2. Research Findings

### 2.1 Basic Pitch Architecture

- **Model:** CNN-based polyphonic pitch detection built on CREPE
- **Framework:** TensorFlow 2.x (also PyTorch port available)
- **Outputs:** Contour (pitch), Note (discrete), Onset (timing)
- **Training Data:** Originally trained on diverse instrument recordings

### 2.2 Fine-Tuning Options

| Approach | Pros | Cons |
|----------|------|------|
| **TensorFlow (original)** | Full compatibility | Complex setup |
| **PyTorch port** | Easier experimentation | Inference only, no training code |
| **Custom training pipeline** | Full control | Build from scratch |

### 2.3 Available Resources

- [basic-pitch-torch](https://github.com/gudgud96/basic-pitch-torch) - PyTorch inference
- [audio_to_midi_transcriber](https://github.com/alexfishy12/audio_to_midi_transcriber) - Custom training example
- [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro) - 200+ hours of piano with aligned MIDI

---

## 3. User Stories

### 3.1 Genre-Specific Model

**As a** jazz musician
**I want** a model trained on jazz piano
**So that** it better recognizes swing timing, blue notes, and jazz voicings

**Acceptance Criteria:**
- [ ] Can select `--model jazz` when extracting
- [ ] Jazz model handles swing quantization correctly
- [ ] Recognizes extended chords (9ths, 11ths, 13ths) better than stock

### 3.2 Custom Training

**As a** producer with my own sample library
**I want** to train on my recordings
**So that** extraction matches my specific sound/style

**Acceptance Criteria:**
- [ ] CLI command to prepare training data: `midi-grep train prepare`
- [ ] CLI command to fine-tune: `midi-grep train run --epochs 50`
- [ ] Trained model saved locally for reuse
- [ ] `--model custom` flag to use trained model

### 3.3 Model Comparison

**As a** user
**I want** to compare stock vs custom model output
**So that** I can verify fine-tuning improved accuracy

**Acceptance Criteria:**
- [ ] `midi-grep compare --models stock,custom --input audio.wav`
- [ ] Shows note count, accuracy diff, timing precision

---

## 4. Technical Approach

### 4.1 Training Data Format

```
training_data/
├── audio/
│   ├── sample_001.wav
│   ├── sample_002.wav
│   └── ...
├── midi/
│   ├── sample_001.mid
│   ├── sample_002.mid
│   └── ...
└── manifest.json
```

### 4.2 Training Pipeline

```bash
# 1. Prepare dataset
midi-grep train prepare --audio-dir ./audio --midi-dir ./midi --output ./dataset

# 2. Fine-tune (transfer learning from Basic Pitch weights)
midi-grep train run \
  --dataset ./dataset \
  --base-model basic-pitch \
  --epochs 100 \
  --batch-size 16 \
  --output ./models/my-jazz-model

# 3. Use custom model
midi-grep extract --url "..." --model ./models/my-jazz-model
```

### 4.3 Pre-Built Genre Models

Distribute pre-trained models for common genres:

| Model | Training Data | Use Case |
|-------|---------------|----------|
| `jazz` | Jazz piano recordings | Swing, extended chords |
| `classical` | MAESTRO dataset | Classical piano, dynamics |
| `electronic` | Synth/EDM samples | Arps, leads, bass |
| `lofi` | Lo-fi hip-hop samples | Chopped/sampled piano |

### 4.4 Model Registry

```bash
# List available models
midi-grep models list

# Download pre-built model
midi-grep models pull jazz

# Share custom model
midi-grep models push my-model --public
```

---

## 5. Implementation Phases

### Phase A: Research & Prototype
- [ ] Set up TensorFlow training environment
- [ ] Download MAESTRO dataset
- [ ] Create minimal training script
- [ ] Verify fine-tuned model works with existing pipeline

### Phase B: CLI Integration
- [ ] Add `midi-grep train` subcommand
- [ ] Add `--model` flag to extract command
- [ ] Model storage in `~/.midi-grep/models/`

### Phase C: Pre-Built Models
- [ ] Train jazz model on curated dataset
- [ ] Train electronic model on synth samples
- [ ] Host models for download

### Phase D: Model Sharing
- [ ] Export/import model format
- [ ] Optional: Public model registry

---

## 6. Dependencies

- TensorFlow 2.x for training
- ~10GB disk space for MAESTRO dataset
- GPU recommended for training (CPU possible but slow)
- Additional Python packages: `tensorflow-io`, `mir_eval`

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Training requires GPU | Provide cloud training option or pre-built models |
| Large dataset downloads | Cache datasets, incremental downloads |
| Model overfitting | Cross-validation, early stopping, data augmentation |
| Complex setup for users | Pre-built models cover 80% of use cases |

---

## 8. Success Metrics

- Fine-tuned model achieves 10%+ accuracy improvement on target genre
- Training completes in <2 hours on consumer GPU
- Users can train custom model with <100 samples
- Pre-built models downloaded 100+ times

---

## 9. References

- [Basic Pitch Paper (ICASSP 2022)](https://arxiv.org/abs/2203.09893)
- [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)
- [PyTorch Port](https://github.com/gudgud96/basic-pitch-torch)
- [Training Example](https://github.com/alexfishy12/audio_to_midi_transcriber)
