from __future__ import annotations
from typing import Tuple

import jax.numpy as jnp
import jax

from pydantic import BaseModel

from .dataset import ImageDataSet
from dm_pix import ssim
from time import time_ns

from scipy.spatial.distance import euclidean

class Loss(BaseModel):   

    @staticmethod
    def dissim_score(
        arctan_img_raw: jnp.array,
        source_img_raw: jnp.array,
        l_threshold: float,
    ) -> Tuple[int, int]:
        ssim_score = ssim(arctan_img_raw, source_img_raw)

        dist_raw = (1.0 - ssim_score) / 2.0
        dist = jnp.maximum(dist_raw - l_threshold, 0.0)

        (dist_raw, dist)

    @staticmethod
    def bottlesim_score(
        target_image_features: jnp.array,
        modded_image_features: jnp.array,
        dissm_score: float,
        modifier: float,
        budget: float       
    ) -> float:

        dist_tfeat_mfeat = euclidean(target_image_features, modded_image_features)
        modified_maximum = modifier * max(dissm_score - budget, 0)

        return dist_tfeat_mfeat + modified_maximum


    @staticmethod
    def compute_fawkes_loss(
        arctan_img_raw: jnp.array,
        source_img_raw: jnp.array,
        target_image_features: jnp.array,
        modded_image_features: jnp.array,
        modifier: float,
        budget: float
    ) -> Tuple[float, float, float]:

        dissim_raw, dissim = Loss.dissim_score(
            arctan_img_raw=arctan_img_raw,
            source_img_raw=source_img_raw
        )

        loss = Loss.bottlesim_score(
            target_image_features=target_image_features,
            modded_image_features=modded_image_features,
            modifier=modifier,
            budget=budget
        )

        return (loss, dissim_raw, dissim)