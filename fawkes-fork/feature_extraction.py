from __future__ import annotations

from typing import List

import jax
import jax.numpy as jnp
from .models import deepid, vggface, facenet, facenet512,deepface, openface, arcface 


models = {
    "ArcFace": arcface.loadArcFace(),
    "DeepFace": deepface.loadDeepFace(),
    "Facenet": facenet.loadFacenet(),
    "FaceNet512": facenet512.loadFaceNet512(),
    "DeepId": deepid.loadDeepId(),
    "VGGFace": vggface.loadVGGFace(),
}



class FeatureExtractor:  
    @jax.jit
    @staticmethod
    def load_feature_reprt(
        image_pixels: jnp.array
    ):  
        reprs = {}

        @jax.jit
        def make_repr(model_name, image_pixels=image_pixels, models=models):
            reprs[model_name] = models[model_name].predict(image_pixels)


        list = [
            "ArcFace"
            "DeepFace"
            "Facenet"
            "FaceNet512"
            "DeepId"
            "VGGFace"
        ]

        jax.vmap(make_repr)(list)

        return reprs


 