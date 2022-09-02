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
import numpy as np
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

        sum = jnp.asarray(jnps[0])[:, :, jnp.newaxis]

        for j in jnps[1:]:
            sum += jnp.asarray(j)[:, :, jnp.newaxis]

        maps_mean = sum / len(jnps)

        return (dssim_score, maps_mean)

    @staticmethod
    def loss_score(
        target_image_features: Iterable,
        modded_image_features: Iterable,
        dssim_map: Iterable,
        modifier: Iterable,
        budget: float
    ) -> float:

        i, j, _ = modded_image_features.shape


        modded_image_features = modded_image_features.flatten()[jnp.newaxis, :]
        target_image_features = (target_image_features + jnp.zeros((i, j, 1))).flatten()[jnp.newaxis, :]

        dist_tfeat_mfeat = ImageModelOps.compare_faces(
            target_image_features, modded_image_features)
      
        modified_maximum = modifier[0] * \
            jnp.max(jnp.asarray([dssim_map - budget, jnp.zeros(dssim_map.shape)]))

        return dist_tfeat_mfeat + modified_maximum

    @staticmethod
    def loss_score_model_dicts(
        params: optax.Params,
        source_image_features: Dict,
        target_image_features: Dict,
        dssim_map: Iterable,
    ) -> Tuple[Iterable, float, float]:

        ret = []

        def single_loss(
            model_name: str,
            a=target_image_features,
            b=source_image_features,
            dssim_map=dssim_map,
            modifier=params['modifier'],
            budget=params['budget']
        ):
            a_feat = a[model_name]
            b_feat = b[model_name]

            ret.append(Loss.loss_score(
                a_feat,
                b_feat,
                dssim_map=dssim_map,
                modifier=modifier,
                budget=budget
            ))

        list = [
            "ArcFace",
            "DeepFace",
            "Facenet",
            "FaceNet512",
            "DeepId",
            "VGGFace",
        ]

        for l in list:
            single_loss(l)

        arr = jnp.asarray(ret)
        mean = jnp.mean(arr)

        return mean
