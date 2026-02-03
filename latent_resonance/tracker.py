"""Face tracking and feature extraction via MediaPipe FaceLandmarker (Tasks API)."""

from __future__ import annotations

import os
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

_BaseOptions = mp.tasks.BaseOptions
_FaceLandmarker = mp.tasks.vision.FaceLandmarker
_FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
_RunningMode = mp.tasks.vision.RunningMode

# Default model URL and cache location
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_CACHE_DIR = Path.home() / ".cache" / "latent_resonance"
_MODEL_FILENAME = "face_landmarker.task"


def _ensure_model(model_path: str | None = None) -> str:
    """Return path to the face landmarker model, downloading if necessary."""
    if model_path and os.path.isfile(model_path):
        return model_path

    cached = _MODEL_CACHE_DIR / _MODEL_FILENAME
    if cached.is_file():
        return str(cached)

    _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading face landmarker model to {cached} ...")
    urllib.request.urlretrieve(_MODEL_URL, cached)
    print("Download complete.")
    return str(cached)


@dataclass
class FacialFeatures:
    """Normalized facial features, all in ``[0, 1]``."""

    jaw_openness: float
    eyebrow_tension: float
    head_yaw: float
    stillness: float


# 3-D model points for solvePnP (canonical face, arbitrary scale).
# Landmarks: nose tip (1), left eye corner (33), right eye corner (263),
#            left mouth (61), right mouth (291), chin (199).
_MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],            # nose tip
    [-30.0, -30.0, -30.0],      # left eye corner
    [30.0, -30.0, -30.0],       # right eye corner
    [-25.0, 30.0, -15.0],       # left mouth corner
    [25.0, 30.0, -15.0],        # right mouth corner
    [0.0, 55.0, -10.0],         # chin
], dtype=np.float64)

_LANDMARK_IDS = [1, 33, 263, 61, 291, 199]

# Velocity sliding window size (frames)
_VELOCITY_WINDOW = 15


class FaceTracker:
    """Extracts facial features from webcam frames via MediaPipe FaceLandmarker.

    Call :meth:`start` to open the webcam, then :meth:`read_features` each
    frame.  Call :meth:`stop` to release resources.
    """

    def __init__(
        self, camera_index: int = 0, model_path: str | None = None,
    ) -> None:
        self._camera_index = camera_index
        self._model_path = model_path
        self._cap: cv2.VideoCapture | None = None
        self._landmarker: _FaceLandmarker | None = None
        self._prev_landmarks: np.ndarray | None = None
        self._velocity_history: deque[float] = deque(maxlen=_VELOCITY_WINDOW)
        self._frame_timestamp_ms: int = 0
        self._debug_frame: np.ndarray | None = None

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {self._camera_index}. "
                "On macOS, grant camera access to your terminal app: "
                "System Settings → Privacy & Security → Camera."
            )

        resolved_model = _ensure_model(self._model_path)
        options = _FaceLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=resolved_model),
            running_mode=_RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = _FaceLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def read_features(self) -> FacialFeatures | None:
        """Capture one frame and return extracted features, or ``None`` if no face."""
        if self._cap is None or self._landmarker is None:
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires monotonically increasing timestamps
        self._frame_timestamp_ms += 33  # ~30 FPS
        result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if not result.face_landmarks:
            return None

        face_lms = result.face_landmarks[0]  # first face
        h, w = frame.shape[:2]

        # Convert to numpy array of pixel coords (x, y, z)
        pts = np.array(
            [(lm.x * w, lm.y * h, lm.z * w) for lm in face_lms],
            dtype=np.float64,
        )

        jaw = self._compute_jaw_openness(pts)
        brow = self._compute_eyebrow_tension(pts)
        yaw = self._compute_head_yaw(pts[:, :2], w, h)
        still = self._compute_stillness(pts)

        features = FacialFeatures(
            jaw_openness=jaw,
            eyebrow_tension=brow,
            head_yaw=yaw,
            stillness=still,
        )
        self._debug_frame = self._draw_debug(frame, pts, features)

        return features

    def get_debug_frame(self) -> np.ndarray | None:
        """Return the last frame with face mesh and feature values drawn on it."""
        return self._debug_frame

    @staticmethod
    def _draw_debug(
        frame: np.ndarray, pts: np.ndarray, features: FacialFeatures,
    ) -> np.ndarray:
        """Draw landmark dots and feature readouts on a copy of the frame."""
        out = frame.copy()

        # Draw all landmarks as small green dots
        for x, y, *_ in pts:
            cv2.circle(out, (int(x), int(y)), 1, (0, 255, 0), -1)

        # Highlight key landmarks used for feature extraction
        # Jaw: upper lip (13), lower lip (14)
        for idx in (13, 14):
            cv2.circle(out, (int(pts[idx][0]), int(pts[idx][1])), 3, (0, 0, 255), -1)
        # Eyebrows/eyes: 46, 159, 276, 386
        for idx in (46, 159, 276, 386):
            cv2.circle(out, (int(pts[idx][0]), int(pts[idx][1])), 3, (255, 0, 0), -1)
        # Head pose anchors
        for idx in _LANDMARK_IDS:
            cv2.circle(out, (int(pts[idx][0]), int(pts[idx][1])), 3, (0, 255, 255), -1)

        # Feature readout
        lines = [
            f"jaw:  {features.jaw_openness:.2f}",
            f"brow: {features.eyebrow_tension:.2f}",
            f"yaw:  {features.head_yaw:.2f}",
            f"still:{features.stillness:.2f}",
        ]
        for i, text in enumerate(lines):
            cv2.putText(
                out, text, (10, 25 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA,
            )

        return out

    # ── Feature computation helpers ──────────────────────────────────────

    @staticmethod
    def _compute_jaw_openness(pts: np.ndarray) -> float:
        """Vertical distance between upper/lower lip, normalized by face height."""
        upper_lip = pts[13]  # landmark 13
        lower_lip = pts[14]  # landmark 14
        lip_dist = np.linalg.norm(upper_lip[:2] - lower_lip[:2])

        # Face height: top of head (10) to chin (152)
        face_top = pts[10]
        chin = pts[152]
        face_height = np.linalg.norm(face_top[:2] - chin[:2])

        if face_height < 1e-6:
            return 0.0

        raw = lip_dist / face_height
        # Typical range ~0.01 (closed) to ~0.15 (wide open)
        return float(np.clip(raw / 0.15, 0.0, 1.0))

    @staticmethod
    def _compute_eyebrow_tension(pts: np.ndarray) -> float:
        """Average brow-to-eye distance, inverted so higher = more tension."""
        # Left brow (46) to left eye (159), right brow (276) to right eye (386)
        left_dist = np.linalg.norm(pts[46][:2] - pts[159][:2])
        right_dist = np.linalg.norm(pts[276][:2] - pts[386][:2])
        avg_dist = (left_dist + right_dist) / 2.0

        # Normalize by face height
        face_height = np.linalg.norm(pts[10][:2] - pts[152][:2])
        if face_height < 1e-6:
            return 0.0

        ratio = avg_dist / face_height
        # Typical range ~0.04 (tense/furrowed) to ~0.08 (relaxed/raised)
        # Invert: small distance = high tension
        tension = 1.0 - np.clip((ratio - 0.04) / 0.04, 0.0, 1.0)
        return float(tension)

    @staticmethod
    def _compute_head_yaw(pts_2d: np.ndarray, w: int, h: int) -> float:
        """Estimate head yaw via solvePnP, mapped to [0, 1] with 0.5 = centered."""
        image_points = pts_2d[_LANDMARK_IDS].astype(np.float64)

        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, _ = cv2.solvePnP(
            _MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.5

        rmat, _ = cv2.Rodrigues(rvec)
        # Extract yaw (Y-axis rotation) from rotation matrix
        yaw_rad = np.arctan2(rmat[2, 0], rmat[2, 2])
        # Map from roughly [-45, +45] degrees to [0, 1]
        yaw_deg = np.degrees(yaw_rad)
        return float(np.clip((yaw_deg + 45.0) / 90.0, 0.0, 1.0))

    def _compute_stillness(self, pts: np.ndarray) -> float:
        """Running average of landmark velocity, inverted (low velocity = high stillness)."""
        if self._prev_landmarks is not None:
            delta = np.linalg.norm(pts - self._prev_landmarks, axis=1).mean()
            self._velocity_history.append(delta)

        self._prev_landmarks = pts.copy()

        if not self._velocity_history:
            return 1.0  # no motion yet → maximum stillness

        avg_velocity = np.mean(self._velocity_history)
        # Typical velocity range: 0 (still) to ~10 (fast movement)
        stillness = 1.0 - np.clip(avg_velocity / 10.0, 0.0, 1.0)
        return float(stillness)
