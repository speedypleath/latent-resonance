"""Generate 5 spectrograms from the latest checkpoint and reconstruct audio via Griffin-Lim."""

import os
import sys
import pickle
import subprocess

import numpy as np
import torch
from PIL import Image

# ── Clone StyleGAN3 for pickle module resolution ─────────────────────────
SG3_DIR = "/tmp/stylegan3"
if not os.path.exists(SG3_DIR):
    subprocess.run(["git", "clone", "--quiet",
                    "https://github.com/NVlabs/stylegan3.git", SG3_DIR], check=True)
sys.path.insert(0, SG3_DIR)

from latent_resonance.dataset import spectrogram_to_audio, save_audio
from latent_resonance.dataset.processing import spectrogram_to_image

# ── Settings ──────────────────────────────────────────────────────────────
CHECKPOINT = "checkpoints/network-snapshot-000400.pkl"
OUTPUT_DIR = "reconstructed_audio"
NUM_SAMPLES = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load generator ────────────────────────────────────────────────────────
print(f"Loading checkpoint: {CHECKPOINT}")
with open(CHECKPOINT, "rb") as f:
    data = pickle.load(f)

G = data["G_ema"]
if torch.cuda.is_available():
    G = G.cuda().eval()
    device = "cuda"
else:
    G = G.cpu().eval()
    device = "cpu"
print(f"Generator loaded (z_dim={G.z_dim}, device={device})")

# ── Generate spectrograms ────────────────────────────────────────────────
z = torch.randn(NUM_SAMPLES, G.z_dim, device=device)
with torch.no_grad():
    imgs = G(z, None)
print(f"Generated {NUM_SAMPLES} spectrograms, shape: {imgs.shape}")

# ── Save spectrograms & reconstruct audio ─────────────────────────────────
for i in range(NUM_SAMPLES):
    img_tensor = imgs[i]  # (C, H, W) in [-1, 1]

    if img_tensor.shape[0] == 3:
        # RGB model (magma colormap) — use the full 3-channel tensor
        spec = img_tensor[0].cpu().numpy()  # scalar channel for audio
        img_pil = spectrogram_to_image(spec)
    else:
        # Legacy 1-channel model
        spec = img_tensor[0].cpu().numpy()
        img_pil = spectrogram_to_image(spec)

    png_path = os.path.join(OUTPUT_DIR, f"sample_{i}.png")
    img_pil.save(png_path)
    print(f"  Sample {i} spectrogram → {png_path}")

    # Reconstruct audio via Griffin-Lim
    audio = spectrogram_to_audio(spec)
    wav_path = os.path.join(OUTPUT_DIR, f"sample_{i}.wav")
    save_audio(audio, wav_path)
    duration = len(audio) / 22050
    print(f"  Sample {i}: {len(audio)} samples ({duration:.2f}s) → {wav_path}")

# ── Also reconstruct one dataset spectrogram for comparison ───────────────
dataset_png = "data/spectrograms/752588_complejo_formado-Dark-pad.png"
print(f"\nReconstructing dataset sample: {dataset_png}")
img = Image.open(dataset_png)
audio = spectrogram_to_audio(img)
wav_path = os.path.join(OUTPUT_DIR, "dataset_sample.wav")
save_audio(audio, wav_path)
print(f"  {len(audio)} samples ({len(audio)/22050:.2f}s) → {wav_path}")

print(f"\nDone — all files saved to {OUTPUT_DIR}/")
