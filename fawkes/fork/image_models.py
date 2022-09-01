from __future__ import annotations
from statistics import mode

from typing import Dict, List, Iterable

import jax
import jax.numpy as jnp
from scipy.spatial.distance import cdist

from .models import (arcface, deepface, deepid, facenet, facenet512, openface,
                     vggface)

import cv2
import numpy as np

models = {
    "ArcFace": arcface.loadArcFace(),
    "DeepFace": deepface.loadDeepFace(),
    "Facenet": facenet.loadFacenet(),
    "FaceNet512": facenet512.loadFaceNet512(),
    "DeepId": deepid.loadDeepId(),
    "VGGFace": vggface.loadVGGFace(),
}


class ImageModelOps:

    @staticmethod
    def load_feature_reprt(
        image_pixels: Iterable
    ):
        reprs = {}

        for k, v in models.items():
            input_shape = v.layers[0].input_shape

            if type(input_shape) == list:
                input_shape = input_shape[0][1:3]
            else:
                input_shape = input_shape[1:3]

            if input_shape == (55, 47):
                input_shape = (47, 55)
            
            image_pixels = cv2.resize(np.asarray(image_pixels), input_shape)

            reprs[k] = v.predict(image_pixels[np.newaxis, :, :, :])

        return reprs

    @staticmethod
    def compare_faces(
        image_a_features: Iterable,
        image_b_features: Iterable,
    ) -> jnp.array:
        return jnp.asarray(
            cdist(
                image_a_features,
                image_b_features
            )
        )

    @staticmethod
    def compare_features_dict_models(
        image_feats_a: Dict,
        image_feats_b: Dict,
    ):
        dists = {}

        for k, _ in models.items():
            a_feat = image_feats_a[k]
            b_feat = image_feats_b[k]

            dists[k] = ImageModelOps.compare_faces(
                a_feat,
                b_feat,
            )

        return dists

    @staticmethod
    def compare_images_and_apply_threshold(
        image_a_features: Iterable,
        image_b_features: Iterable,
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
    def compare_feature_apply_thresh_dict(
        dict_feat_a: Dict,
        dict_feat_b: Dict,
        threshold: float
    ) -> Dict[str, bool]:
        res = {}

        for k, _ in models.items():
            a_feat = dict_feat_a[k]
            b_feat = dict_feat_b[k]

            res[k] = ImageModelOps.compare_images_and_apply_threshold(
                a_feat,
                b_feat,
                threshold
            )

        return res
