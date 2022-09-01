from __future__ import annotations

from re import S
from time import time_ns
from typing import Dict, Tuple, Iterable

import jax
import jax.numpy as jnp
import optax
from dm_pix import ssim
from fawkes import get_dissim_map_and_sim_score
from pydantic import BaseModel

from .image_models import ImageModelOps


class Loss(BaseModel):

    @staticmethod
    def dissim_map_and_score(
        source_img_rctan: Iterable,
        target_img_rctan: Iterable,
    ) -> Tuple[float, jnp.array]:
        dssim_score, maps = get_dissim_map_and_sim_score(
            source_img_rctan,
            target_img_rctan
        )

        maps_jnp = sum([[jnp.asarray(t[1])
                        for t in tup]
                        for tup in maps
                        ], [])

        maps_mean = jnp.mean(jnp.asarray(maps_jnp))

        return (dssim_score, maps_mean)

    @staticmethod
    def loss_score(
        target_image_features: Iterable,
        modded_image_features: Iterable,
        dssim_map: Iterable,
        modifier: Iterable,
        budget: float
    ) -> float:

        dist_tfeat_mfeat = ImageModelOps.compare_faces(
            target_image_features, modded_image_features)
        modified_maximum = modifier * \
            jnp.max(dssim_map - budget, jnp.zeros(dssim_map.shape))

        return dist_tfeat_mfeat + modified_maximum

    @staticmethod
    def loss_score_model_dicts(
        params: optax.Params,
        target_image_features: Dict,
        dssim_map: Iterable,
        i: int,
    ) -> Tuple[Iterable, float, float]:

        ret = {}

        @jax.jit
        def single_loss(
            model_name: str,
            a=target_image_features,
            b=params['best_results'][i],
            dssim_map=dssim_map,
            modifier=params['modifier'],
            budget=params['budget']
        ):
            a_feat = a[model_name]
            b_feat = b[model_name]

            ret[model_name] = Loss.loss_score(
                a_feat,
                b_feat,
                dssim_map=dssim_map,
                modifier=modifier,
                budget=budget
            )

        list = [
            "ArcFace",
            "DeepFace",
            "Facenet",
            "FaceNet512",
            "DeepId",
            "VGGFace",
        ]

        jax.vmap(single_loss)(list)

        arr = jnp.asarray(list(ret.values()))
        mean = jnp.mean(arr)

        return mean
