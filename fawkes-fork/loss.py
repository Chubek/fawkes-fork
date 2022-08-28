from __future__ import annotations

import jax.numpy as jnp
import jax

from pydantic import BaseModel

from .dataset import ImageDataSet


class Loss(BaseModel):
    aimg_raw_batch: ImageDataSet = ()
    simg_raw_batch: ImageDataSet = ()
    aimg_input_batct: ImageDataSet = ()
    timg_input_batch: ImageDataSet = ()
    simg_input_batch: ImageDataSet = ()
