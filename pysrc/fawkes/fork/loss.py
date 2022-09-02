from __future__ import annotations

from re import S
from time import time_ns
from typing import Dict, Iterable, Tuple

import jax
import jax.numpy as jnp
import optax
from dm_pix import ssim
from fawkes_ext import get_dissim_map_and_sim_score
from pydantic import BaseModel

from .image_models import ImageModelOps


class Loss(BaseModel):

    @staticmethod
    def dissim_map_and_score(
        source_img_tanh: Iterable,
        target_img_tanh: Iterable,
    ) -> Tuple[float, jnp.array]:
        dssim_score, maps = get_dissim_map_and_sim_score(
            source_img_tanh.astype(jnp.uint8).tolist(),
            target_img_tanh.astype(jnp.uint8).tolist()
        )

        jnps = []
        base_shape = None

        for i, m in enumerate(maps):
            arr = jnp.asarray(m)
            
            if i == 0:
                base_shape = arr.shape
                jnps.append(arr)
                continue

            i, j = arr.shape
            ii, jj = base_shape

            i_pad = (ii - i)
            j_pad = (jj - j)

            arr = jnp.pad(arr,((j_pad // 2, j_pad // 2 + j_pad % 2), 
                     (i_pad // 2, i_pad // 2 + j_pad % 2)),
                  mode = 'constant')

            jnps.append(arr)

        maps_mean = jnp.mean(jnp.asarray(jnps))

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
