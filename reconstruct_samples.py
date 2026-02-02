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
    spec = imgs[i, 0].cpu().numpy()  # (H, W) in [-1, 1]

    # Save spectrogram as PNG (grayscale, flipped to match forward pipeline)
    img_uint8 = ((spec + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8, mode="L").transpose(Image.FLIP_TOP_BOTTOM)
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
