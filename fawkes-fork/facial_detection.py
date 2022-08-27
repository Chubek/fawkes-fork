from __future__ import annotations

from enum import Enum
from os import stat
from typing import List, Tuple
import jax.numpy as jnp
import jax
from deepface.DeepFace import detectFace


class FacialDetector(Enum):
    OpenCV = "opencv"
    MTCNN = "mtcnn"
    SSD = "ssd"
    Dlib = "dlib"
    Mediapipe = "mediapipe"
    RetinaFace = "retinaface"

    @staticmethod
    def from_str(s: str) -> FacialDetector:
        s = s.lower()

        if s == "opencv":
            return FacialDetector.OpenCV
        elif s == "mtcnn":
            return FacialDetector.MTCNN
        elif s == "ssd":
            return FacialDetector.SSD
        elif s == "dlib":
            return FacialDetector.Dlib
        elif s == "mediapipe":
            return FacialDetector.Mediapipe
        elif s == "retinaface":
            return FacialDetector.RetinaFace

    @staticmethod
    def to_string(fdet: FacialDetector) -> str:
        if fdet == FacialDetector.OpenCV:
            return "OpenCV"
        elif fdet == FacialDetector.MTCNN:
            return "MTCNN"
        elif fdet == FacialDetector.SSD:
            return "SSD"
        elif fdet == FacialDetector.Dlib:
            return "Dlib"
        elif fdet == FacialDetector.Mediapipe:
            return "Mediapipe"
        elif fdet == FacialDetector.RetinaFace:
            return "RetinaFace"


    @staticmethod
    @jax.jit
    def detect_face(
        img_path: str,
        model: FacialDetector,
        target_size: Tuple[int, int]
    ) -> jnp.array:
        detector_backend = FacialDetector.to_string(model)

        detected_face = detectFace(
            img_path=img_path,
            detector_backend=detector_backend,
            target_size=target_size
        )

        return jnp.asarray(detected_face)


    @staticmethod
    @jax.jit
    def detect_face_batch(
        batch: List[str],
        model: FacialDetector,
        target_size: Tuple[int, int]
    ) -> jnp.array:
        return jax.vmap(FacialDetector.detect_face)(batch, model, target_size)

