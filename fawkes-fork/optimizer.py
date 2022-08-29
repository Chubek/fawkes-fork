from __future__ import annotations

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

    source_images: ImageDataSet = ImageDataSet()
    target_images: ImageDataSet = ImageDataSet()


    @classmethod
    def new(cls, lr: float) -> Optimizer:
        obj = cls()
        obj.lr = lr


        return obj

    

    
