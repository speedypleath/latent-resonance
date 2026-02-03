"""Generate spectrograms from checkpoint(s) and reconstruct audio via Griffin-Lim.

Supports a single checkpoint file or a training-run directory containing
multiple ``network-snapshot-*.pkl`` files.  When given a directory, each
checkpoint gets its own subfolder with independently sampled latent vectors.
"""

import glob
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
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    "checkpoints/training-runs/"
    "00000-stylegan2-spectrograms-gpus2-batch16-gamma2",
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "reconstructed_audio")
NUM_SAMPLES = 5
TRUNCATION_PSI = 0.5

# ── Discover checkpoints ─────────────────────────────────────────────────
if os.path.isdir(CHECKPOINT_PATH):
    checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_PATH, "network-snapshot-*.pkl")))
    if not checkpoints:
        sys.exit(f"No network-snapshot-*.pkl files found in {CHECKPOINT_PATH}")
    print(f"Found {len(checkpoints)} checkpoints in {CHECKPOINT_PATH}")
else:
    checkpoints = [CHECKPOINT_PATH]

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

for ckpt_path in checkpoints:
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]  # e.g. network-snapshot-000040
    ckpt_tag = ckpt_name.replace("network-snapshot-", "kimg")       # e.g. kimg000040
    ckpt_out = os.path.join(OUTPUT_DIR, ckpt_tag)
    os.makedirs(ckpt_out, exist_ok=True)

    # ── Load generator ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Loading checkpoint: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        data = pickle.load(f)

    G = data["G_ema"]
    G = G.cuda().eval() if device == "cuda" else G.cpu().eval()
    print(f"Generator loaded (z_dim={G.z_dim}, img_channels={G.img_channels}, device={device})")

    z = torch.randn(NUM_SAMPLES, G.z_dim, device=device)

    # ── Generate spectrograms ─────────────────────────────────────────────
    with torch.no_grad():
        imgs = G(z, None, truncation_psi=TRUNCATION_PSI)
    print(f"Generated {NUM_SAMPLES} spectrograms, shape: {imgs.shape}")

    # ── Save spectrograms & reconstruct audio ─────────────────────────────
    for i in range(NUM_SAMPLES):
        raw = imgs[i, 0].cpu().numpy()  # (H, W) in [-1, 1], already in PNG orientation

        # Save spectrogram as grayscale PNG — GAN output already matches
        # the flipped orientation of the training PNGs, no extra flip needed.
        img_uint8 = ((raw + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8, mode="L")
        png_path = os.path.join(ckpt_out, f"sample_{i}.png")
        img_pil.save(png_path)
        print(f"  Sample {i} spectrogram → {png_path}")

        # Undo the vertical flip for audio reconstruction — spectrogram_to_audio
        # expects the raw numpy layout (low freq at row 0).
        spec = raw[::-1].copy()
        audio = spectrogram_to_audio(spec)
        wav_path = os.path.join(ckpt_out, f"sample_{i}.wav")
        save_audio(audio, wav_path)
        duration = len(audio) / 22050
        print(f"  Sample {i}: {len(audio)} samples ({duration:.2f}s) → {wav_path}")

    # Free model memory before loading next checkpoint
    del G, data

print(f"\nDone — all files saved to {OUTPUT_DIR}/")
