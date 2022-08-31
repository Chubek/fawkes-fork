from __future__ import annotations
from typing import Tuple

import jax.numpy as jnp
import jax

from fawkes import get_dissim_map_and_sim_score


from pydantic import BaseModel

from dm_pix import ssim
from time import time_ns

from scipy.spatial.distance import euclidean

class Loss(BaseModel):   

    @staticmethod
    def dissim_map_and_score(
        source_img_raw: jnp.array,
        target_img_raw: jnp.array,
    ) -> Tuple[float, jnp.array]:
        dssim_score, maps = get_dissim_map_and_sim_score(source_img_raw, target_img_raw)

        maps_jnp = sum([[jnp.asarray(t[1])
                        for t in tup]
                        for tup in maps
                            ], [])

        maps_mean = jnp.mean(jnp.asarray(maps_jnp))

        return (dssim_score, maps_mean)

    @staticmethod
    def loss_score(
        target_image_features: jnp.array,
        modded_image_features: jnp.array,
        dssim_map: jnp.array,
        modifier: jnp.array,
        budget: float       
    ) -> float:

        dist_tfeat_mfeat = euclidean(target_image_features, modded_image_features)
        modified_maximum = modifier * jnp.max(dssim_map - budget, jnp.zeros(dssim_map.shape))

        return dist_tfeat_mfeat + modified_maximum


 