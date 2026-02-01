# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Latent Resonance: Facial Acoustics** — a Python generative system that converts facial micro-movements into audio via GAN-generated spectrograms. The user's face is never displayed; instead, facial data drives sound synthesis parameters.

## Setup and Running

```bash
# Install dependencies and create virtual environment
uv sync

# Add a new dependency
uv add <package>

# Run (requires webcam and model weights in /checkpoints/model.pt)
uv run python main.py
```

Requires Python 3.13+, [uv](https://docs.astral.sh/uv/), and a webcam.

## Dataset Processing

Convert audio files to 512x512 grayscale spectrogram PNGs for GAN training:

```bash
# CLI entry point
uv run latent-resonance-dataset ./raw_audio ./data/spectrograms

# Module invocation
uv run python -m latent_resonance.dataset ./raw_audio ./data/spectrograms

# With custom parameters
uv run latent-resonance-dataset ./raw_audio ./data/spectrograms --sr 22050 --n-mels 512 --hop-length 512 --n-fft 2048
```

Programmatic usage:

```python
from latent_resonance.dataset import process_directory, audio_to_spectrogram, SpectrogramDataset

# Process a whole directory
process_directory("./raw_audio", "./data/spectrograms")

# Load as PyTorch dataset (tensors in [-1, 1], shape (1, 512, 512))
dataset = SpectrogramDataset("./data/spectrograms")
```

Supported audio formats: `.wav`, `.mp3`, `.flac`, `.ogg`

## Freesound Scraper

Search [freesound.org](https://freesound.org) and download audio previews (HQ OGG ~192kbps) for GAN training data:

```bash
# CLI (API key via flag)
uv run latent-resonance-scraper "cello" ./raw_audio --api-key YOUR_KEY --num-results 100

# CLI (API key via env var)
export FREESOUND_API_KEY=YOUR_KEY
uv run latent-resonance-scraper "cello sustained" ./raw_audio --min-duration 5 --max-duration 30

# Choose mp3 format instead of ogg
uv run latent-resonance-scraper "cello" ./raw_audio --format mp3
```

Programmatic usage:

```python
from latent_resonance.dataset import scrape_freesound

scrape_freesound("cello", "./raw_audio", api_key="...", num_results=50)
```

Requires a free Freesound API key from https://freesound.org/apiv2/apply/.

## Architecture

Four-stage pipeline:

1. **Input** — OpenCV + MediaPipe tracks 468 facial landmarks from webcam
2. **Translation** — Extracts spatiotemporal features (velocity, jaw openness, head tilt), maps to latent vector `z` with linear interpolation smoothing
3. **Generation** — GAN produces 512x512 spectrogram images from latent vector
4. **Reconstruction** — Griffin-Lim algorithm (librosa) estimates phase from spectrogram amplitude to produce audio

### Control Mapping

Facial actions map to energy states, not 1:1 coordinates:
- Head rotation (Y-axis angular velocity) → time-stretch/playback speed
- Mouth openness (lip distance) → bandpass filter frequency range
- Eyebrow tension (brow-eye distance) → noise/entropy in latent vector
- Stillness (near-zero delta position) → clarity/pure tone

### Key Technical Decisions

- **Griffin-Lim** for phase estimation: spectrograms store only amplitude, so phase must be reconstructed iteratively
- **Linear interpolation** on input vector: prevents chaotic flashing from raw coordinate mapping, creates fluid drone-like morphing
- **Log-scale dB normalization** to [-1, 1] range for GAN training stability
- Structured harmonic source material (cello, speech) trains better than noisy/white-noise spectrograms

## Project Structure

```
latent_resonance/
    __init__.py
    dataset/
        __init__.py          # re-exports: process_directory, audio_to_spectrogram, SpectrogramDataset, scrape_freesound
        processing.py        # core pipeline (audio → spectrogram → PNG)
        dataset.py           # PyTorch SpectrogramDataset class
        cli.py               # argparse CLI entry point
        __main__.py          # shim for python -m latent_resonance.dataset
        scraper.py           # freesound.org search + download
```
