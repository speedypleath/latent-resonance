from pathlib import Path
import warnings

import librosa
import numpy as np
from PIL import Image

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 512
IMAGE_SIZE = 512

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


def audio_to_spectrogram(
    audio_path: str | Path,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
) -> np.ndarray:
    """Convert an audio file to a normalized mel spectrogram.

    Returns an ndarray of shape (n_mels, T) with values in [-1, 1].
    """
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    eps = 1e-8
    s_min = S_db.min()
    s_max = S_db.max()
    normalized = 2.0 * (S_db - s_min) / (s_max - s_min + eps) - 1.0
    return normalized


def spectrogram_to_image(
    spectrogram: np.ndarray, size: int = IMAGE_SIZE
) -> Image.Image:
    """Convert a [-1, 1] spectrogram array to a grayscale PIL Image."""
    pixel_values = ((spectrogram + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(pixel_values, mode="L")
    img = img.resize((size, size), Image.LANCZOS)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def process_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
) -> list[Path]:
    """Process all audio files in input_dir, saving spectrogram PNGs to output_dir."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(
        f for f in input_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not audio_files:
        warnings.warn(f"No supported audio files found in {input_dir}")
        return []

    saved: list[Path] = []
    for i, audio_path in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] Processing {audio_path.name}...")
        try:
            spec = audio_to_spectrogram(
                audio_path, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
            )
            img = spectrogram_to_image(spec)
            out_path = output_dir / f"{audio_path.stem}.png"
            img.save(out_path)
            saved.append(out_path)
        except Exception as e:
            warnings.warn(f"Skipping {audio_path.name}: {e}")

    print(f"Done. {len(saved)}/{len(audio_files)} files processed.")
    return saved
