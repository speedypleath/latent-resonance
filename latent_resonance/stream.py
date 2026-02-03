"""Real-time audio output via sounddevice with crossfading between buffers."""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
import sounddevice as sd

# Crossfade length in samples (~93 ms at 22050 Hz)
_CROSSFADE_LEN = 2048


def _raised_cosine(n: int) -> np.ndarray:
    """Raised-cosine window of length *n*, rising from 0 to 1."""
    return (1.0 - np.cos(np.linspace(0, np.pi, n))) / 2.0


class AudioStream:
    """Streams audio chunks to the speakers with crossfading.

    New audio is enqueued from the generation thread via :meth:`enqueue`.
    When the queue is empty the current buffer loops with a crossfade at the
    boundary to avoid clicks (drone sustain).

    Parameters
    ----------
    sr : int
        Sample rate.
    max_queue : int
        Maximum number of pending audio buffers.
    """

    def __init__(self, sr: int = 22050, max_queue: int = 3) -> None:
        self._sr = sr
        self._queue: deque[np.ndarray] = deque(maxlen=max_queue)
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None

        # Playback state (accessed only from the callback thread)
        self._current: np.ndarray | None = None
        self._pos: int = 0

        # Pre-compute crossfade windows
        self._fade_in = _raised_cosine(_CROSSFADE_LEN).astype(np.float32)
        self._fade_out = self._fade_in[::-1].copy()

    def start(self) -> None:
        self._stream = sd.OutputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            callback=self._callback,
            blocksize=1024,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def enqueue(self, audio: np.ndarray) -> None:
        """Add an audio buffer to the playback queue (thread-safe)."""
        buf = np.asarray(audio, dtype=np.float32).ravel()
        with self._lock:
            self._queue.append(buf)

    # ── sounddevice callback ─────────────────────────────────────────────

    def _callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        needed = frames
        out = np.zeros(needed, dtype=np.float32)
        written = 0

        while written < needed:
            if self._current is None or len(self._current) == 0:
                buf = self._try_dequeue_latest()
                if buf is not None:
                    self._current = buf
                    self._pos = 0
                else:
                    break

            remaining_in_buf = len(self._current) - self._pos
            chunk_size = min(needed - written, remaining_in_buf)
            out[written:written + chunk_size] = self._current[self._pos:self._pos + chunk_size]
            self._pos += chunk_size
            written += chunk_size

            if self._pos >= len(self._current):
                next_buf = self._try_dequeue_latest()
                if next_buf is not None:
                    # Crossfade old tail into new head
                    self._apply_crossfade(out, written, next_buf)
                    self._current = next_buf
                    self._pos = min(_CROSSFADE_LEN, len(next_buf))
                else:
                    # Loop: crossfade tail into head of same buffer
                    self._apply_crossfade(out, written, self._current)
                    self._pos = min(_CROSSFADE_LEN, len(self._current))

        outdata[:] = out.reshape(-1, 1)

    def _try_dequeue_latest(self) -> np.ndarray | None:
        """Return the newest queued buffer, discarding stale ones."""
        with self._lock:
            if not self._queue:
                return None
            while len(self._queue) > 1:
                self._queue.popleft()
            return self._queue.popleft()

    def _apply_crossfade(
        self, out: np.ndarray, write_pos: int, incoming: np.ndarray,
    ) -> None:
        """Overlap-add crossfade: fade out the tail in *out*, fade in head of *incoming*."""
        n = min(_CROSSFADE_LEN, write_pos, len(incoming))
        if n <= 0:
            return
        start = write_pos - n
        if n == _CROSSFADE_LEN:
            fade_out = self._fade_out
            fade_in = self._fade_in
        else:
            fade_in = _raised_cosine(n).astype(np.float32)
            fade_out = fade_in[::-1]
        out[start:write_pos] = out[start:write_pos] * fade_out + incoming[:n] * fade_in
