"""Live webcam → face tracking → GAN → audio pipeline.

The user's face is never displayed.  Instead, facial micro-movements drive
latent-space navigation in a trained StyleGAN, producing spectrograms that are
reconstructed into audio in real time.

Press **q** to quit.
"""

import os
import queue
import signal
import sys
import threading

import cv2
import matplotlib
import numpy as np

from latent_resonance import (
    AudioStream,
    FaceTracker,
    LatentMapper,
    SpectrogramGenerator,
)
from latent_resonance.dataset import spectrogram_to_audio

# ── Settings ─────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    "checkpoints/training-runs/"
    "00000-stylegan2-spectrograms-gpus2-batch16-gamma2/"
    "network-snapshot-000040.pkl",
)
TRUNCATION_PSI = float(os.environ.get("TRUNCATION_PSI", "0.5"))
SAMPLE_RATE = 22050
GRIFFIN_LIM_ITERS = 16  # fewer iterations for real-time speed

# ── Magma colormap for display ───────────────────────────────────────────────

_cmap = matplotlib.colormaps["magma"]
_MAGMA_LUT = (_cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


def _spectrogram_to_display(spec: np.ndarray) -> np.ndarray:
    """Convert a [-1, 1] spectrogram to a BGR image for cv2.imshow."""
    indices = ((spec + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    # Flip so low frequencies are at the bottom
    indices = indices[::-1]
    rgb = _MAGMA_LUT[indices]  # (H, W, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ── Generation thread ────────────────────────────────────────────────────────

def _generation_loop(
    generator: SpectrogramGenerator,
    audio_stream: AudioStream,
    z_queue: queue.Queue,
    display_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """Background thread: consume latent vectors, generate spectrograms, reconstruct audio."""
    while not stop_event.is_set():
        # Drain the queue to get the latest z (discard stale ones)
        z = None
        try:
            while True:
                z = z_queue.get_nowait()
        except queue.Empty:
            pass

        if z is None:
            # Nothing new — wait briefly
            stop_event.wait(0.05)
            continue

        # Generate spectrogram (already in natural orientation: row 0 = low freq)
        spec = generator.generate(z)

        audio = spectrogram_to_audio(spec, n_iter=GRIFFIN_LIM_ITERS)
        audio_stream.enqueue(audio)

        # Send display image
        display_img = _spectrogram_to_display(spec)
        try:
            # Drain old frames and put the latest
            while not display_queue.empty():
                display_queue.get_nowait()
        except queue.Empty:
            pass
        display_queue.put(display_img)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading GAN checkpoint: {CHECKPOINT_PATH}")
    if not os.path.isfile(CHECKPOINT_PATH):
        sys.exit(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Set the CHECKPOINT_PATH environment variable to the .pkl file."
        )

    generator = SpectrogramGenerator(
        CHECKPOINT_PATH, truncation_psi=TRUNCATION_PSI,
    )
    print(f"Generator loaded (z_dim={generator.z_dim})")

    tracker = FaceTracker()
    mapper = LatentMapper(z_dim=generator.z_dim)
    audio_stream = AudioStream(sr=SAMPLE_RATE)

    z_queue: queue.Queue = queue.Queue(maxsize=8)
    display_queue: queue.Queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    # Graceful shutdown on Ctrl+C
    def _signal_handler(sig: int, frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)

    # Start subsystems
    tracker.start()
    audio_stream.start()

    gen_thread = threading.Thread(
        target=_generation_loop,
        args=(generator, audio_stream, z_queue, display_queue, stop_event),
        daemon=True,
    )
    gen_thread.start()

    print("Pipeline running. Press 'q' to quit.")

    try:
        while not stop_event.is_set():
            features = tracker.read_features()
            if features is not None:
                z = mapper(features)
                try:
                    z_queue.put_nowait(z)
                except queue.Full:
                    pass  # drop if generation can't keep up

            # Show face mesh debug view
            debug_frame = tracker.get_debug_frame()
            if debug_frame is not None:
                cv2.imshow("Face Mesh Debug", debug_frame)

            # Show latest spectrogram
            try:
                img = display_queue.get_nowait()
                cv2.imshow("Latent Resonance", img)
            except queue.Empty:
                pass

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        stop_event.set()
        gen_thread.join(timeout=3.0)
        audio_stream.stop()
        tracker.stop()
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
