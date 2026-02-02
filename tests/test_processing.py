"""Tests for the spectrogram forward & inverse pipeline."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from PIL import Image

from latent_resonance.dataset.processing import (
    audio_to_spectrogram,
    save_audio,
    spectrogram_to_audio,
    spectrogram_to_image,
)

SR = 22050


@pytest.fixture()
def sine_wave() -> np.ndarray:
    """1-second 440 Hz sine wave."""
    t = np.linspace(0, 1, SR, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture()
def audio_file(tmp_path: Path, sine_wave: np.ndarray) -> Path:
    """Write the sine wave to a temporary WAV file."""
    path = tmp_path / "sine.wav"
    sf.write(str(path), sine_wave, SR)
    return path


@pytest.fixture()
def spectrogram(audio_file: Path) -> np.ndarray:
    """Mel spectrogram array produced by the forward pipeline."""
    return audio_to_spectrogram(audio_file)


# ── Forward Pipeline ─────────────────────────────────────────────────────────


def test_audio_to_spectrogram_shape(spectrogram: np.ndarray) -> None:
    assert spectrogram.shape[0] == 512
    assert spectrogram.shape[1] > 0


def test_audio_to_spectrogram_range(spectrogram: np.ndarray) -> None:
    assert spectrogram.min() >= -1.0
    assert spectrogram.max() <= 1.0


def test_spectrogram_to_image_size(spectrogram: np.ndarray) -> None:
    img = spectrogram_to_image(spectrogram)
    assert img.size == (512, 512)


def test_spectrogram_to_image_mode(spectrogram: np.ndarray) -> None:
    img = spectrogram_to_image(spectrogram)
    assert img.mode == "L"


def test_spectrogram_to_image_pixel_range(spectrogram: np.ndarray) -> None:
    img = spectrogram_to_image(spectrogram)
    pixels = np.array(img)
    assert pixels.min() >= 0
    assert pixels.max() <= 255


# ── Inverse Pipeline ─────────────────────────────────────────────────────────


def test_spectrogram_to_audio_from_array(spectrogram: np.ndarray) -> None:
    audio = spectrogram_to_audio(spectrogram)
    assert audio.ndim == 1
    assert np.any(audio != 0)


def test_spectrogram_to_audio_from_image(spectrogram: np.ndarray) -> None:
    img = spectrogram_to_image(spectrogram)
    audio = spectrogram_to_audio(img)
    assert audio.ndim == 1
    assert np.any(audio != 0)


def test_spectrogram_to_audio_from_tensor(spectrogram: np.ndarray) -> None:
    tensor = torch.from_numpy(spectrogram).unsqueeze(0)  # (1, H, W)
    audio = spectrogram_to_audio(tensor)
    assert audio.ndim == 1
    assert np.any(audio != 0)


def test_spectrogram_to_audio_rejects_3d() -> None:
    bad = np.zeros((2, 512, 512), dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        spectrogram_to_audio(bad)


def test_save_audio_creates_file(
    tmp_path: Path, spectrogram: np.ndarray
) -> None:
    audio = spectrogram_to_audio(spectrogram)
    out = tmp_path / "out.wav"
    result = save_audio(audio, out)
    assert result.exists()
    data, sr = sf.read(str(result))
    assert len(data) > 0


def test_save_audio_creates_parents(
    tmp_path: Path, spectrogram: np.ndarray
) -> None:
    audio = spectrogram_to_audio(spectrogram)
    out = tmp_path / "a" / "b" / "out.wav"
    result = save_audio(audio, out)
    assert result.exists()


# ── Round-Trip ────────────────────────────────────────────────────────────────


def test_round_trip_array(spectrogram: np.ndarray) -> None:
    audio = spectrogram_to_audio(spectrogram)
    assert audio.ndim == 1
    assert np.any(audio != 0)


def test_round_trip_via_image(spectrogram: np.ndarray) -> None:
    img = spectrogram_to_image(spectrogram)
    audio = spectrogram_to_audio(img)
    assert audio.ndim == 1
    assert np.any(audio != 0)


# ── Real Data Tests ──────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AUDIO_DIR = DATA_DIR / "audio"
SPECTROGRAM_DIR = DATA_DIR / "spectrograms"

_has_real_data = AUDIO_DIR.is_dir() and any(AUDIO_DIR.iterdir())

real_data = pytest.mark.skipif(
    not _has_real_data, reason="No real audio data in data/audio/"
)


def _sample_audio_files(n: int = 5) -> list[Path]:
    """Return up to *n* audio files from data/audio/, evenly spaced."""
    if not AUDIO_DIR.is_dir():
        return []
    files = sorted(AUDIO_DIR.iterdir())
    if len(files) <= n:
        return files
    step = len(files) // n
    return [files[i * step] for i in range(n)]


def _matching_spectrogram(audio_path: Path) -> Path:
    return SPECTROGRAM_DIR / f"{audio_path.stem}.png"


@real_data
@pytest.mark.parametrize(
    "audio_path", _sample_audio_files(), ids=lambda p: p.name
)
class TestRealForward:
    """Forward pipeline tests against real OGG files from data/audio/."""

    def test_spectrogram_shape(self, audio_path: Path) -> None:
        spec = audio_to_spectrogram(audio_path)
        assert spec.shape[0] == 512
        assert spec.shape[1] > 0

    def test_spectrogram_range(self, audio_path: Path) -> None:
        spec = audio_to_spectrogram(audio_path)
        assert spec.min() >= -1.0
        assert spec.max() <= 1.0

    def test_image_matches_saved(self, audio_path: Path) -> None:
        """Generated image should be close to the one already on disk."""
        saved = _matching_spectrogram(audio_path)
        if not saved.exists():
            pytest.skip(f"No saved spectrogram for {audio_path.name}")
        spec = audio_to_spectrogram(audio_path)
        generated = spectrogram_to_image(spec)
        saved_img = Image.open(saved)
        gen_arr = np.array(generated, dtype=np.float64)
        saved_arr = np.array(saved_img, dtype=np.float64)
        mae = np.abs(gen_arr - saved_arr).mean()
        assert mae < 1.0, f"Mean absolute pixel error {mae:.2f} >= 1.0"


@real_data
@pytest.mark.parametrize(
    "audio_path", _sample_audio_files(), ids=lambda p: p.name
)
class TestRealInverse:
    """Inverse pipeline tests: spectrogram PNG -> audio reconstruction."""

    def test_image_to_audio(self, audio_path: Path) -> None:
        saved = _matching_spectrogram(audio_path)
        if not saved.exists():
            pytest.skip(f"No saved spectrogram for {audio_path.name}")
        img = Image.open(saved)
        audio = spectrogram_to_audio(img)
        assert audio.ndim == 1
        assert len(audio) > 0
        assert np.any(audio != 0)

    def test_round_trip_nonzero(self, audio_path: Path) -> None:
        spec = audio_to_spectrogram(audio_path)
        audio = spectrogram_to_audio(spec)
        assert audio.ndim == 1
        assert np.any(audio != 0)

    def test_save_round_trip(self, audio_path: Path, tmp_path: Path) -> None:
        spec = audio_to_spectrogram(audio_path)
        audio = spectrogram_to_audio(spec)
        out = tmp_path / f"{audio_path.stem}_rt.wav"
        result = save_audio(audio, out)
        assert result.exists()
        data, sr = sf.read(str(result))
        assert len(data) > 0
        assert sr == SR
