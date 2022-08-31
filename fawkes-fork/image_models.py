from __future__ import annotations
from tkinter import Image

from typing import Dict, List

import jax
import jax.numpy as jnp
from .models import deepid, vggface, facenet, facenet512, deepface, openface, arcface
from scipy.spatial.distance import cdist

models = {
    "ArcFace": arcface.loadArcFace(),
    "DeepFace": deepface.loadDeepFace(),
    "Facenet": facenet.loadFacenet(),
    "FaceNet512": facenet512.loadFaceNet512(),
    "DeepId": deepid.loadDeepId(),
    "VGGFace": vggface.loadVGGFace(),
}


class ImageModelOps:
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
            "ArcFace",
            "DeepFace",
            "Facenet",
            "FaceNet512",
            "DeepId",
            "VGGFace",
        ]

        jax.vmap(make_repr)(list)

        return reprs

    @jax.jit
    @staticmethod
    def compare_faces(
        image_a_features: jnp.array,
        image_b_features: jnp.array,
    ) -> jnp.array:
        return jnp.asarray(
            cdist(
                image_a_features,
                image_b_features
            )
        )

    @jax.jit
    @staticmethod
    def compare_features_dict_models(
        image_feats_a: Dict,
        image_feats_b: Dict,
    ):
        dists = {}

        @jax.jit
        def get_distance(
            model_name: str,
            a=image_feats_a,
            b=image_feats_b
        ):
            a_feat = a[model_name]
            b_feat = b[model_name]

            dists[model_name] = ImageModelOps.compare_faces(
                a_feat,
                b_feat,
            )

        list = [
            "ArcFace",
            "DeepFace",
            "Facenet",
            "FaceNet512",
            "DeepId",
            "VGGFace",
        ]

        jax.vmap(get_distance)(list)

        return dists

    @staticmethod
    def compare_images_and_apply_threshold(
        image_a_features: jnp.array,
        image_b_features: jnp.array,
        threshold: float,
    ) -> bool:
        dist_map = ImageModelOps.compare_faces(
            image_a_features,
            image_b_features
        )

        avg = jnp.mean(dist_map)

        if avg <= threshold:
            return True

        return False

    @staticmethod
    @jax.jit
    def compare_feature_apply_thresh_dict(
        dict_feat_a: Dict,
        dict_feat_b: Dict,
        threshold: float
    ) -> Dict[str, bool]:
        res = {}

        @jax.jit
        def compare_feat(
            key: str,
            a=dict_feat_a,
            b=dict_feat_b,
            thresh=threshold,
        ):
            a_feat = a[key]
            b_feat = b[key]

            res[key] = ImageModelOps.compare_images_and_apply_threshold(
                a_feat,
                b_feat,
                thresh
            )

        list = [
            "ArcFace",
            "DeepFace",
            "Facenet",
            "FaceNet512",
            "DeepId",
            "VGGFace",
        ]

        jax.vmap(compare_feat)(list)

        return res
