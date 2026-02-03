from pathlib import Path
import warnings

import librosa
import matplotlib
import numpy as np
import scipy.ndimage
import scipy.signal
import soundfile as sf
from PIL import Image

try:
    import torch

    _has_torch = True
except ImportError:
    _has_torch = False

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 512
IMAGE_SIZE = 512

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}

# ── Magma colormap lookup tables ─────────────────────────────────────────────
_cmap = matplotlib.colormaps["magma"]
_MAGMA_LUT = (_cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)  # (256, 3)

# Perceived luminance (BT.601) for each LUT entry – used for RGB→scalar inversion
_MAGMA_LUMINANCE = (
    0.299 * _MAGMA_LUT[:, 0].astype(np.float64)
    + 0.587 * _MAGMA_LUT[:, 1].astype(np.float64)
    + 0.114 * _MAGMA_LUT[:, 2].astype(np.float64)
)

# Pre-sorted luminance for searchsorted inversion
_LUMA_ORDER = np.argsort(_MAGMA_LUMINANCE)
_LUMA_SORTED = _MAGMA_LUMINANCE[_LUMA_ORDER]


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
    """Convert a [-1, 1] spectrogram array to an RGB PIL Image using the magma colormap."""
    indices = ((spectrogram + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    rgb = _MAGMA_LUT[indices]  # (H, W, 3)
    img = Image.fromarray(rgb, mode="RGB")
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


def _minimum_phase_init(S_mag: np.ndarray, n_fft: int) -> np.ndarray:
    """Compute a minimum-phase STFT estimate from a magnitude spectrogram.

    Uses the cepstral method: take the log-magnitude, inverse-DFT to get the
    cepstrum, window to keep only the causal half, then DFT back to obtain a
    complex spectrum whose phase is the minimum-phase estimate.

    Parameters
    ----------
    S_mag : np.ndarray, shape (n_freq, n_frames)
        Non-negative magnitude spectrogram (amplitude, not power).
    n_fft : int
        FFT size used to produce *S_mag*.

    Returns
    -------
    np.ndarray, shape (n_freq, n_frames), complex64
        Complex STFT with minimum-phase estimate and original magnitudes.
    """
    eps = 1e-10
    log_mag = np.log(np.maximum(S_mag, eps))

    # Inverse DFT along frequency axis to get cepstrum
    cepstrum = np.fft.irfft(log_mag, n=n_fft, axis=0)

    # Causal window: keep DC, double causal part, zero anti-causal
    n_freq = S_mag.shape[0]  # n_fft // 2 + 1
    window = np.zeros(n_fft, dtype=cepstrum.dtype)
    window[0] = 1.0
    window[1: n_fft // 2] = 2.0
    if n_fft % 2 == 0:
        window[n_fft // 2] = 1.0
    cepstrum *= window[:, np.newaxis]

    # Back to frequency domain for minimum-phase spectrum
    min_phase_spec = np.fft.rfft(cepstrum, n=n_fft, axis=0)
    # Extract phase, apply to original magnitude
    phase = np.exp(1j * np.angle(np.exp(min_phase_spec)))
    return (S_mag * phase).astype(np.complex64)


def _griffinlim_with_init(
    S_mag: np.ndarray,
    S_complex_init: np.ndarray,
    *,
    n_iter: int = 64,
    hop_length: int = HOP_LENGTH,
    n_fft: int = N_FFT,
    momentum: float = 0.99,
) -> np.ndarray:
    """Griffin-Lim algorithm with a custom initial complex STFT.

    Parameters
    ----------
    S_mag : np.ndarray, shape (n_freq, n_frames)
        Target magnitude spectrogram (amplitude).
    S_complex_init : np.ndarray, shape (n_freq, n_frames), complex
        Initial complex STFT estimate (e.g. from minimum-phase).
    n_iter : int
        Number of Griffin-Lim iterations.
    hop_length : int
        STFT hop length.
    n_fft : int
        FFT size.
    momentum : float
        Momentum for faster convergence (0 = classic GL).

    Returns
    -------
    np.ndarray, 1-D float32
        Reconstructed audio waveform.
    """
    S_prev = np.zeros_like(S_complex_init)
    S_complex = S_complex_init.copy()

    for _ in range(n_iter):
        audio = librosa.istft(S_complex, hop_length=hop_length, n_fft=n_fft)
        S_new = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        # Truncate or pad to match target frame count
        n_frames = S_mag.shape[1]
        if S_new.shape[1] > n_frames:
            S_new = S_new[:, :n_frames]
        elif S_new.shape[1] < n_frames:
            pad = np.zeros(
                (S_new.shape[0], n_frames - S_new.shape[1]),
                dtype=S_new.dtype,
            )
            S_new = np.concatenate([S_new, pad], axis=1)

        phase = np.exp(1j * np.angle(S_new))
        S_update = S_mag * phase
        S_complex = S_update + momentum * (S_update - S_prev)
        S_prev = S_update

    return librosa.istft(S_complex, hop_length=hop_length, n_fft=n_fft).astype(
        np.float32
    )


def _rgb_to_scalar(rgb: np.ndarray) -> np.ndarray:
    """Convert an (H, W, 3) RGB magma image (0-255 float) to a [-1, 1] scalar array.

    For each pixel, compute BT.601 luminance, find the closest magma LUT entry,
    and map the LUT index back to [-1, 1].
    """
    luma = (
        0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    )
    # Find nearest LUT entry via searchsorted on pre-sorted luminance
    insert_idx = np.searchsorted(_LUMA_SORTED, luma.ravel()).clip(0, 255)
    # Map sorted position back to original LUT index
    lut_indices = _LUMA_ORDER[insert_idx].reshape(luma.shape)
    # LUT index 0..255 → scalar -1..1
    return lut_indices.astype(np.float32) / 255.0 * 2.0 - 1.0


def spectrogram_to_audio(
    spectrogram: "np.ndarray | Image.Image",
    *,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    n_iter: int = 64,
    db_range: float = 80.0,
    lpf_cutoff: float = 8000.0,
    spectral_smooth_sigma: float = 0.5,
    use_min_phase: bool = True,
    noise_floor_db: float = -60.0,
) -> np.ndarray:
    """Reconstruct audio waveform from a [-1,1] mel spectrogram via Griffin-Lim.

    Accepts a numpy array in [-1, 1], a PIL Image (as saved by the forward
    pipeline), or a PyTorch tensor of shape ``(1, H, W)`` or ``(H, W)``.

    Returns a 1-D numpy array of audio samples at sample rate *sr*.
    """
    # --- normalise input to numpy [-1, 1] of shape (H, W) ----------------
    if isinstance(spectrogram, Image.Image):
        # Undo the vertical flip applied in spectrogram_to_image
        spectrogram = spectrogram.transpose(Image.FLIP_TOP_BOTTOM)
        arr = np.array(spectrogram, dtype=np.float32)
        if arr.ndim == 3:
            # RGB magma image → invert via luminance
            arr = _rgb_to_scalar(arr)
        else:
            # Legacy grayscale
            arr = arr / 255.0 * 2.0 - 1.0
    elif _has_torch and isinstance(spectrogram, torch.Tensor):
        arr = spectrogram.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            # (3, H, W) RGB tensor → invert via luminance
            arr = _rgb_to_scalar(np.transpose(arr, (1, 2, 0)) * 127.5 + 127.5)
        else:
            arr = arr.squeeze()  # (1, H, W) or (H, W)
    else:
        arr = np.asarray(spectrogram, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(
            f"Expected a 2-D spectrogram, got shape {arr.shape}"
        )

    # --- [-1, 1] → dB -------------------------------------------------
    S_db = (arr + 1.0) * (db_range / 2.0) - db_range

    # --- Spectral smoothing (reduce 8-bit quantisation artifacts) ------
    if spectral_smooth_sigma > 0:
        S_db = scipy.ndimage.gaussian_filter(
            S_db, sigma=[spectral_smooth_sigma, spectral_smooth_sigma * 0.5]
        )

    # --- dB → mel power -----------------------------------------------
    S_mel = librosa.db_to_power(S_db, ref=1.0)

    # --- Noise floor suppression --------------------------------------
    if noise_floor_db < 0:
        peak_power = S_mel.max()
        threshold = peak_power * librosa.db_to_power(noise_floor_db, ref=1.0)
        S_mel = np.where(S_mel < threshold, 0.0, S_mel)

    # --- mel → STFT magnitude -----------------------------------------
    S_stft = librosa.feature.inverse.mel_to_stft(
        S_mel, sr=sr, n_fft=n_fft, power=2.0,
    )

    # --- Griffin-Lim reconstruction -----------------------------------
    S_amp = np.sqrt(np.maximum(S_stft, 0.0))
    if use_min_phase:
        S_init = _minimum_phase_init(S_amp, n_fft)
        audio = _griffinlim_with_init(
            S_amp, S_init,
            n_iter=n_iter, hop_length=hop_length,
            n_fft=n_fft, momentum=0.99,
        )
    else:
        audio = librosa.griffinlim(
            S_stft, n_iter=n_iter, hop_length=hop_length,
            momentum=0.99, n_fft=n_fft,
        )

    # --- Low-pass filter (8 kHz default, order 4) ---------------------
    if lpf_cutoff and lpf_cutoff < sr / 2:
        sos = scipy.signal.butter(
            4, lpf_cutoff, btype="low", fs=sr, output="sos",
        )
        audio = scipy.signal.sosfiltfilt(sos, audio).astype(np.float32)

    # --- Boundary fades (5 ms raised-cosine to eliminate clicks) ------
    fade_samples = int(0.005 * sr)
    if len(audio) > 2 * fade_samples:
        fade_in = (1.0 - np.cos(np.linspace(0, np.pi, fade_samples))) / 2.0
        fade_out = (1.0 - np.cos(np.linspace(np.pi, 0, fade_samples))) / 2.0
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out

    # The forward pipeline normalises dB relative to the signal's peak power
    # (ref=np.max), so the absolute level is lost.  Normalise the
    # reconstructed waveform to a -1 dBFS peak to restore a usable amplitude.
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 10 ** (-1.0 / 20.0)  # ≈ 0.891

    return audio


def save_audio(
    audio: np.ndarray,
    output_path: "str | Path",
    sr: int = SAMPLE_RATE,
) -> Path:
    """Save audio waveform to a WAV file.

    Returns the resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sr)
    return output_path
