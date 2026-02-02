# Latent Resonance: Facial Acoustics

**A computational instrument that transmutes facial micro-movements into generated audio spectrograms.**

## Project Overview

This project is a generative software system that interprets spatiotemporal data from the human face to navigate a machine-learning model trained on audio spectrograms. By combining **computer vision** with **generative audio synthesis**, it creates a feedback loop where the user performs "silent vocals" that the machine interprets into new, hallucinatory sounds.

**Core Concept:** The system adheres to the constraint of **non-representation**. The user’s body is never visualized. Instead, their physical presence is abstracted into the *parameters* of a sound wave—frequency, amplitude, and noise—visualized as a spectrogram and reconstructed into audio in real-time.

## System Architecture

The pipeline consists of four distinct stages:

1. **Input (Computer Vision):** A webcam captures the user. `MediaPipe` tracks 468 facial landmarks.
2. **Translation (Regression):** Spatiotemporal features (velocity, jaw openness, head tilt) are extracted and mapped to a high-dimensional latent vector ($z$).
3. **Generation (The Model):** A GAN (Generative Adversarial Network) generates a unique $512 \times 512$ spectrogram image based on the input vector.
4. **Reconstruction (Sonification):** The generated image is converted into audio using the **Griffin-Lim algorithm** (via `librosa`), estimating phase data to produce sound.

## The Dataset

**Source Material:**
The model was trained on a custom-curated dataset of almost 500 spectrograms generated from audio samples sourced via [freesound.org](https://freesound.org) (cello, sustained tones, reese bass, and similar structured harmonic material).

**Curation Process:**

* **Audio Sourcing:** Samples were downloaded using the built-in freesound scraper (`latent-resonance-scraper`), filtering by duration and selecting sounds with clear harmonic structure.
* **Audio Processing:** Raw audio was batch-processed using `librosa.feature.melspectrogram`, producing 512x512 grayscale PNGs.
* **Normalization:** All spectrograms were converted to log-scale (dB) and normalized to a range of `[-1, 1]` to stabilize GAN training.
* **Learnings:** I discovered that "noisy" images (like white noise spectrograms) confuse the model, while structured harmonic sounds (like cello or speech) produce cleaner visual patterns.

## Training the GAN

The curated spectrogram dataset was used to train a StyleGAN model using transfer learning from pre-trained weights (FFHQ). Training was performed on free-tier GPU platforms (Google Colab and Kaggle) with significant modifications to accommodate hardware limitations (detailed in the Technical Challenges section).

### Phase 1: The Nightmare Phase (0 – 50 kimg)

**Time**: First hour.

**Visual**: You will see terrifying, melted faces. Some might have "bassline textures" instead of skin, but you will clearly see eyes or mouths.

![alt text](assets/image.png)

**Audio Quality**: Unusable. It will sound like static with weird voice-like formants.

### Phase 2: The "Structure" Phase (50 – 200 kimg)

**Time**: Hours 1 to 4.

**Visual**: The faces fade away. You start seeing horizontal lines (which are bass notes).

**Audio Quality**: "Lo-Fi". The audio will be recognizable as a drone or bass, but it will sound "watery" or "phasey" (like a low-quality MP3). The background silence might be hissy because the blacks aren't fully black yet.

### Phase 3: The "Usable" Phase (200 – 400 kimg)

**Time**: Hours 4 to 8. (Target this window!)

**Visual**: The lines become sharp and high-contrast. The background becomes solid black.

**Audio Quality**: Good. The sharper the image, the clearer the tone. Sharp lines = Pure sine waves. Blurry lines = Noise.

**Action**: If you see distinct, sharp horizontal bands in your fakes.png, STOP. You are done.

## Controls & Interaction

The system does not map 1:1 coordinates (e.g., "head left = sound left"). Instead, it maps **energy states**:

| Facial Action | Spatiotemporal Data | Sonic/Visual Result |
| :--- | :--- | :--- |
| **Head Rotation (Y-Axis)** | Angular Velocity | **Time-Stretch:** Controls the playback speed of the generated sample. |
| **Mouth Openness** | Distance (LipTop, LipBottom) | **Bandpass Filter:** Determines the frequency range (Mouth closed = Low pass; Open = Full spectrum). |
| **Eyebrow Tension** | Distance (Brow, Eye) | **Noise/Entropy:** Adds jitter to the latent vector, introducing digital glitches/static. |
| **Stillness** | $\Delta$ Position $\approx 0$ | **Clarity:** The model settles on a "pure" tone. |

## Installation & Usage

**Prerequisites:**

* Python 3.13+
* [uv](https://docs.astral.sh/uv/) package manager
* Webcam

**Setup:**

```bash
# Install dependencies and create virtual environment
uv sync
```

**Running the System:**

1. Clone the repository.
2. Ensure your trained model weights (`model.pt`) are in the `/checkpoints` folder.
3. Run the main script:

```bash
uv run python main.py
```

**Running Tests:**

```bash
# Install dev dependencies (pytest)
uv sync --group dev

# Run the full test suite
uv run pytest tests/ -v
```

## Technical Challenges & Learnings

### 1. The "Phase" Problem

**Challenge**: Standard spectrogram images only store amplitude (loudness) information, discarding phase (timing). This meant my initial generated sounds were "muddy" and lacked definition.

**Solution**: I implemented the Griffin-Lim algorithm to iteratively estimate the phase signal. While computationally expensive for real-time use, it provided the necessary clarity to make the sounds recognizable.

### 2. Latent Space Mapping

**Challenge**: Mapping facial coordinates (x,y,z) directly to GAN inputs often resulted in chaotic, flashing images.

**Solution**: I used a smoothing function (Linear Interpolation) on the input vector. This allows the sound to "morph" slowly rather than snapping instantly, creating a fluid, drone-like aesthetic.

### 3. Training the GAN: Trial and Error

I chose **StyleGAN3-t** (translation-equivariant) for its suitability with spectrograms, where the horizontal axis represents time and translation equivariance preserves temporal structure.

**Colab attempt (single T4):** My first attempt was on Google Colab with a single Tesla T4 GPU. StyleGAN3's custom CUDA ops (`bias_act`, `upfirdn2d`, `filtered_lrelu`) failed to compile — the fused kernels expect a build environment that Colab doesn't provide out of the box. I had to patch the ops to fall back to native PyTorch reference implementations, which are significantly more memory-hungry. This forced me to drastically reduce the model: `cbase=8192`, `cmax=128`, `batch=2` — a quarter of the default capacity — just to fit in the T4's 15 GB VRAM. Training ran for 1000 kimg but Colab's session time limits made long runs unreliable.

**PyTorch compatibility patches:** StyleGAN3's codebase hasn't been updated for recent PyTorch versions. I had to manually patch two breaking changes: `InfiniteSampler` passing `dataset` to the `Sampler` superclass (removed in PyTorch 2.4), and Adam optimizer receiving integer betas `[0, 0.99]` instead of floats (rejected since PyTorch 2.9).

**Kaggle attempt (T4 x2):** To get longer uninterrupted training, I moved to Kaggle which offers dual T4 GPUs and longer session limits. I initially configured multi-GPU training (`--gpus=2`) but ran into issues and fell back to single-GPU mode. The same CUDA ops compilation problem occurred, requiring the same native fallback and reduced model capacity. I trained for 5000 kimg on Kaggle, which produced usable results.

**What I learned:** Free-tier GPU platforms are viable for GAN training but require significant adaptation. The gap between "paper configurations" and what actually runs on a T4 with native ops is large — I ended up using roughly 1/4 of the model's intended capacity. Despite this, the 435-spectrogram dataset was small enough that the reduced model could still learn meaningful structure. Structured harmonic source material (cello, sustained tones) trained noticeably better than percussive or noisy samples.

Training notebooks are available in the `notebooks/` folder: `train_stylegan3.ipynb` (Colab) and `train_stylegan3_kaggle.ipynb` (Kaggle).

## Project Structure

```text
latent_resonance/          # Python package
    __init__.py
    dataset/
        __init__.py        # re-exports public API
        processing.py      # audio ↔ spectrogram conversion
        dataset.py         # PyTorch SpectrogramDataset
        cli.py             # argparse CLI entry point
        __main__.py        # python -m latent_resonance.dataset
        scraper.py         # freesound.org search + download
tests/
    test_processing.py     # Forward/inverse pipeline & round-trip tests
notebooks/                 # Training notebooks
    train_stylegan3.ipynb          # Google Colab version
    train_stylegan3_kaggle.ipynb   # Kaggle version
data/                      # Dataset files (git-ignored)
    audio/                 # Source audio files (OGG)
    spectrograms/          # Generated spectrogram PNGs
    spectrograms.zip       # Packaged dataset for training
```

## Credits

* Computer Vision: Google MediaPipe
* Audio Processing: Librosa
* Model Architecture: StyleGAN3-t (NVIDIA)
* Audio samples: [freesound.org](https://freesound.org)
