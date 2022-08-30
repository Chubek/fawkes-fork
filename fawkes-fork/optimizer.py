from __future__ import annotations
from typing import Any, List

import jax
import jax.numpy as jnp
from optax import adabelief

from .loss import Loss
from .dataset import ImageDataSet

from pydantic import BaseModel

from time import time_ns

class Optimizer(BaseModel):
    lr: float = jax.random.uniform(
        jax.random.PRNGKey(time_ns()), minval=1e-10, maxval=1e-2)
    optimizer: Any
    source_images: ImageDataSet = ImageDataSet()
    target_images: ImageDataSet = ImageDataSet()
    num_imgs: int = 0
    best_bottlesim: List[float] = []
    best_modded_images: jnp.array = []
    initial_const: int = jax.random.randint(
        jax.random.PRNGKey(time_ns()),
        minval=1,
        maxval=10
    )
    modifier: jnp.array = jnp.array([]) 
    const_jnp: jnp.array = jnp.array([])
    const_diff_jnp: jnp.array = jnp.array([])
    


    @classmethod
    def new(
        cls, 
        lr: float, 
        source_images: ImageDataSet,
        target_images: ImageDataSet,
    ) -> Optimizer:
        obj = cls()
        obj.lr = lr
        obj.optimizer = adabelief(obj.lr)
        obj.source_images = source_images
        obj.target_images = target_images
        obj.num_imgs = len(source_images)
        obj.best_bottlesim = [jnp.inf] * obj.num_imgs
        obj.best_modded_images = ImageDataSet.modded_from_arrays(
            shape=source_images[0].img_data.shape,
            len=len(source_images)
        )
        obj.tanh_images = ImageDataSet.arctanned_from_arrays(
            [
                i.img_data 
                    for i in 
                        obj.source_images
        
        ])
        obj.modifier = jax.random.uniform(
            jax.random.PRNGKey(time_ns()),
            shape=tuple([len(source_images)] + source_images[0].img_data.shape)
        ) * 1e-4
        obj.const_jnp = jnp.ones(len(source_images)) * obj.initial_const
        obj.const_diff_jnp = jnp.ones(len(source_images)) * 1.0

        return obj

    
    @jax.jit
    def one_round(self):
        @jax.jit
        def loss_step(opt_state, **kwg):
            pass

