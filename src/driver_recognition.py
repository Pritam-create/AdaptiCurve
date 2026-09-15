"""Standalone YuNet and SFace driver-recognition utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YUNET_MODEL_PATH = (
    PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
)
DEFAULT_SFACE_MODEL_PATH = (
    PROJECT_ROOT / "models" / "face_recognition_sface_2021dec.onnx"
)

# Initial OpenCV SFace LFW reference threshold. Calibrate this with
# project-specific driver data before treating it as a deployment threshold.
DEFAULT_COSINE_THRESHOLD = 0.363


@dataclass
class EmbeddingResult:
    """Result of extracting one face embedding from a BGR frame."""

    status: str
    embedding: Optional[np.ndarray] = None


@dataclass
class RecognitionResult:
    """Best stored-driver match for a query embedding."""

    driver_id: Optional[str]
    similarity: Optional[float]
    recognized: bool


class DriverRecognizer:
    """Load YuNet/SFace once and provide reusable embedding operations."""

    def __init__(
        self,
        yunet_model_path: Path | str = DEFAULT_YUNET_MODEL_PATH,
        sface_model_path: Path | str = DEFAULT_SFACE_MODEL_PATH,
        cosine_threshold: float = DEFAULT_COSINE_THRESHOLD,
        detector_input_size: tuple[int, int] = (320, 320),
    ) -> None:
        self.yunet_model_path = Path(yunet_model_path)
        self.sface_model_path = Path(sface_model_path)
        self.cosine_threshold = cosine_threshold
        self.detector_input_size = detector_input_size

        self._validate_model_path(self.yunet_model_path)
        self._validate_model_path(self.sface_model_path)
        self._validate_opencv_api()

        # Load both models once; they are reused for all later frames.
        self.face_detector = cv2.FaceDetectorYN.create(
            str(self.yunet_model_path),
            "",
            detector_input_size,
        )
        self.face_recognizer = cv2.FaceRecognizerSF.create(
            str(self.sface_model_path),
            "",
        )

        if self.face_detector is None or self.face_recognizer is None:
            raise RuntimeError("Could not initialize the YuNet or SFace model.")

    @staticmethod
    def _validate_model_path(model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Required model file was not found: {model_path}")

    @staticmethod
    def _validate_opencv_api() -> None:
        required_apis = (
            "FaceDetectorYN",
            "FaceRecognizerSF",
            "FaceRecognizerSF_FR_COSINE",
        )
        missing_apis = [api for api in required_apis if not hasattr(cv2, api)]
        if missing_apis:
            raise RuntimeError(
                "This OpenCV installation does not provide: "
                + ", ".join(missing_apis)
            )

    @staticmethod
    def _is_valid_frame(frame: object) -> bool:
        return (
            isinstance(frame, np.ndarray)
            and frame.size > 0
            and frame.ndim == 3
            and frame.shape[2] == 3
        )

    def extract_embedding(self, frame: np.ndarray) -> EmbeddingResult:
        """Detect exactly one BGR face and return its SFace embedding."""

        if not self._is_valid_frame(frame):
            return EmbeddingResult(status="invalid_frame")

        height, width = frame.shape[:2]

        try:
            # YuNet detection returns face boxes and the five alignment keypoints.
            self.face_detector.setInputSize((width, height))
            _, faces = self.face_detector.detect(frame)
        except cv2.error:
            return EmbeddingResult(status="detection_error")

        face_count = 0 if faces is None else len(faces)
        if face_count == 0:
            return EmbeddingResult(status="no_face")
        if face_count > 1:
            return EmbeddingResult(status="multiple_faces")

        try:
            # SFace aligns/crops from the YuNet box/keypoints before embedding.
            aligned_face = self.face_recognizer.alignCrop(frame, faces[0])
            embedding = self.face_recognizer.feature(aligned_face)
        except cv2.error:
            return EmbeddingResult(status="embedding_error")

        return EmbeddingResult(status="ok", embedding=embedding)

    def compare_embeddings(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """Return cosine similarity for two SFace embeddings (higher is closer)."""

        if not isinstance(embedding1, np.ndarray) or not isinstance(embedding2, np.ndarray):
            raise TypeError("Both embeddings must be NumPy arrays.")

        return float(
            self.face_recognizer.match(
                embedding1,
                embedding2,
                cv2.FaceRecognizerSF_FR_COSINE,
            )
        )

    def recognize(
        self,
        query_embedding: np.ndarray,
        stored_embeddings: Mapping[str, np.ndarray],
    ) -> RecognitionResult:
        """Return the highest stored match only when it clears the threshold."""

        best_driver_id: Optional[str] = None
        best_similarity: Optional[float] = None

        for driver_id, stored_embedding in stored_embeddings.items():
            similarity = self.compare_embeddings(query_embedding, stored_embedding)
            if best_similarity is None or similarity > best_similarity:
                best_driver_id = driver_id
                best_similarity = similarity

        if (
            best_driver_id is not None
            and best_similarity is not None
            and best_similarity >= self.cosine_threshold
        ):
            return RecognitionResult(
                driver_id=best_driver_id,
                similarity=best_similarity,
                recognized=True,
            )

        return RecognitionResult(
            driver_id=None,
            similarity=best_similarity,
            recognized=False,
        )
