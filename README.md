# Latent Resonance: Facial Acoustics

**A computational instrument that transmutes facial micro-movements into generated audio spectrograms.**

### 📌 Project Overview
This project is a generative software system that interprets spatiotemporal data from the human face to navigate a machine-learning model trained on audio spectrograms. By combining **computer vision** with **generative audio synthesis**, it creates a feedback loop where the user performs "silent vocals" that the machine interprets into new, hallucinatory sounds.

**Core Concept:** The system adheres to the constraint of **non-representation**. The user’s body is never visualized. Instead, their physical presence is abstracted into the *parameters* of a sound wave—frequency, amplitude, and noise—visualized as a spectrogram and reconstructed into audio in real-time.

---

### 🛠 System Architecture

The pipeline consists of four distinct stages:

1.  **Input (Computer Vision):** A webcam captures the user. `MediaPipe` tracks 468 facial landmarks.
2.  **Translation (Regression):** Spatiotemporal features (velocity, jaw openness, head tilt) are extracted and mapped to a high-dimensional latent vector ($z$).
3.  **Generation (The Model):** A GAN (Generative Adversarial Network) generates a unique $512 \times 512$ spectrogram image based on the input vector.
4.  **Reconstruction (Sonification):** The generated image is converted into audio using the **Griffin-Lim algorithm** (via `librosa`), estimating phase data to produce sound.

---

### 📂 The Dataset: "Visualizing the Invisible"

**Source Material:**
The model was trained on a custom-curated dataset of [Insert Number, e.g., 1,000] spectrograms generated from [Insert Source, e.g., Bird Calls / Industrial Machinery / Human Speech].

**Curation Process:**
* **Audio Processing:** Raw audio was batch-processed using `librosa.feature.melspectrogram`.
* **Normalization:** All spectrograms were converted to log-scale (dB) and normalized to a range of `[-1, 1]` to stabilize GAN training.
* **Learnings:** I discovered that "noisy" images (like white noise spectrograms) confuse the model, while structured harmonic sounds (like cello or speech) produce cleaner visual patterns.

---

### 🎮 Controls & Interaction

The system does not map 1:1 coordinates (e.g., "head left = sound left"). Instead, it maps **energy states**:

| Facial Action | Spatiotemporal Data | Sonic/Visual Result |
| :--- | :--- | :--- |
| **Head Rotation (Y-Axis)** | Angular Velocity | **Time-Stretch:** Controls the playback speed of the generated sample. |
| **Mouth Openness** | Distance (LipTop, LipBottom) | **Bandpass Filter:** Determines the frequency range (Mouth closed = Low pass; Open = Full spectrum). |
| **Eyebrow Tension** | Distance (Brow, Eye) | **Noise/Entropy:** Adds jitter to the latent vector, introducing digital glitches/static. |
| **Stillness** | $\Delta$ Position $\approx 0$ | **Clarity:** The model settles on a "pure" tone. |

---

### 🚀 Installation & Usage

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

### 🧠 Technical Challenges & Learnings

1. The "Phase" Problem

    Challenge: Standard spectrogram images only store amplitude (loudness) information, discarding phase (timing). This meant my initial generated sounds were "muddy" and lacked definition.

    Solution: I implemented the Griffin-Lim algorithm to iteratively estimate the phase signal. While computationally expensive for real-time use, it provided the necessary clarity to make the sounds recognizable.

2. Latent Space Mapping

    Challenge: Mapping facial coordinates (x,y,z) directly to GAN inputs often resulted in chaotic, flashing images.

    Solution: I used a smoothing function (Linear Interpolation) on the input vector. This allows the sound to "morph" slowly rather than snapping instantly, creating a fluid, drone-like aesthetic.

### 📜 Credits

* Computer Vision: Google MediaPipe
* Audio Processing: Librosa
* Model Architecture: [Insert Model Name, e.g., StyleGAN2-ADA / DCGAN]
* Curated by: [Your Name]