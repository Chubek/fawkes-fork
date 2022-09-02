from __future__ import annotations

from copy import deepcopy
from time import time_ns
from typing import Any, Dict, List, Optional, Iterable

import jax
import jax.numpy as jnp
from optax import adabelief, apply_updates
from pydantic import BaseModel

from .face_op import FaceBase
from .loss import Loss


class Optimizer(BaseModel):
    lr: float = jax.random.uniform(
        jax.random.PRNGKey(time_ns()), minval=1e-10, maxval=1e-2)
    optimizer: Any = None
    source_images: List[FaceBase] = []
    target_images: List[FaceBase] = []
    modifier: Iterable = jnp.array([])
    budget: Iterable = jnp.array([])
    best_results: Iterable = jnp.array([])
    params: Dict = {}
    opt_state: Any = None
    num_img: int = 0

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def new(
        cls,
        source_images: List[FaceBase],
        target_images: List[FaceBase],
        lr: Optional[float] = None,
    ) -> Optimizer:
        obj = cls()
        obj.lr = lr
        obj.optimizer = adabelief(obj.lr)
        obj.source_images = source_images
        obj.target_images = target_images
        
        shape = tuple([len(source_images)] + list(source_images[0].face_cropped.shape))

        obj.modifier = jax.random.uniform(
            jax.random.PRNGKey(time_ns()),
            shape=shape) * 1e-4
        obj.budget = jnp.ones(len(source_images))
        obj.best_results = [
            si.feature_repr
            for si in
            source_images
        ]

        obj.params = {
            "modifier": obj.modifier,
            "budget": obj.budget,
            "best_results": obj.best_results
        }

        obj.opt_state = obj.optimizer.init(obj.params)

        obj.num_img = len(source_images)

        return obj

    def one_round(self):
        for i, tup in enumerate(zip(self.source_images, self.target_images)):
            simg, timg = tup
            
            simg_tanh = simg.face_cropped_tanh
            timg_tanh = timg.face_cropped_tanh

            simg_tanh = simg_tanh + self.params['modifier'][i]

            _, maps_mean = Loss.dissim_map_and_score(
                simg_tanh,
                timg_tanh
            )

            self.source_images[i].protected_face = simg.face_cropped + maps_mean

            new_modded_feature = deepcopy(simg.feat_repr)

            for k, v in new_modded_feature.items():
                new_modded_feature[k] = v + maps_mean

            self.params['best_results'][i] = new_modded_feature

            gradient = jax.jacrev(Loss.loss_score_model_dicts)(
                self.params,
                timg.feat_repr,
                maps_mean,
                i
            )

            updates, self.opt_state = self.optimizer.update(
                gradient,
                self.opt_state,
                self.params
            )

            self.params = apply_updates(self.params, updates)

    def fit(self, max_iter=100):
        for _ in range(max_iter):
            self.one_round()
