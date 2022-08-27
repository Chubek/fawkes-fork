from __future__ import annotations

from enum import Enum
from typing import List

import deepface
import jax
import jax.numpy as jnp

from .facial_detection import FacialDetector
from .image_handling import ImageHandler
from deepface.DeepFace import represent

class FeatureExtractor(str, Enum):
    VGGFace = "vggface"
    Facenet = "facenet"
    OpenFace = "openface"
    Dlib = "dlib"
    ArcFace = "arcface"
    SFace = "sface"
    DeepID = "deepid"
    DeepFace = "deepface"
    Facenet512 = "facenet512"

    @staticmethod
    def from_str(s: str) -> FeatureExtractor:
        s = s.lower()

        if s == "vggface":
            return FeatureExtractor.VGGFace
        elif s == "facenet":
            return FeatureExtractor.Facenet
        elif s == "openface":
            return FeatureExtractor.OpenFace        
        elif s == "arcface":
            return FeatureExtractor.ArcFace
        elif s == "dlib":
            return FeatureExtractor.Dlib
        elif s == "sface":
            return FeatureExtractor.SFace
        elif s == "deepid":
            return FeatureExtractor.DeepID
        elif s == "deepface":
            return FeatureExtractor.DeepFace
        elif s == "facenet512":
            return FeatureExtractor.Facenet512
        else:
            raise ValueError("Unsupported extractor backend")

    

    @staticmethod
    def to_string(feature_ext: FeatureExtractor) -> str:
        if feature_ext == FeatureExtractor.VGGFace:
            return "VGG-Face"
        elif feature_ext == FeatureExtractor.Facenet:
            return "Facenet"
        elif feature_ext == FeatureExtractor.OpenFace:
            return "OpenFace"
        elif feature_ext == FeatureExtractor.Dlib:
            return "Dlib"
        elif feature_ext == FeatureExtractor.ArcFace:
            return "ArcFace"
        elif feature_ext == FeatureExtractor.SFace:
            return "SFace"
        elif feature_ext == FeatureExtractor.DeepID:
            return "DeepID"
        elif feature_ext == FeatureExtractor.DeepFace:
            return "DeepFace"
        elif feature_ext == FeatureExtractor.Facenet512:
            return "Facenet512"

    @staticmethod
    @jax.jit
    def extract_feature_repr(
            img_path: str, 
            model: FeatureExtractor,
            detector: FacialDetector         
        ) -> jnp.array:

        model_name = FeatureExtractor.to_string(model)
        detector_backend = FacialDetector.to_string(detector)

        feature_space = represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend
        )

        return jnp.array(feature_space)


    @staticmethod
    @jax.jit
    def extract_feature_batch(
        batch: List[str],
        model: FeatureExtractor,
        detector: FacialDetector
    ) -> jnp.array:
        return jax.vmap(FeatureExtractor.extract_feature_repr)(batch, model, detector)