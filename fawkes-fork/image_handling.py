import glob
import os
from typing import List, Tuple

import cv2
import jax
import jax.numpy as jnp
from pydantic import BaseModel


class ImageHandler:

    @staticmethod
    @jax.jit
    def load_image_batch(
        batch: List[str],
        resize: Tuple[int, int]
    ) -> List[jnp.array]:
        def load_single_image(path: str, resize=resize) -> jnp.array:
            img_read = cv2.imread(path)
            img_read = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)

            return cv2.resize(img_read, resize)

        return jax.vmap(load_single_image)(batch)

    @staticmethod
    @jax.jit
    def tanh_image_batch(
        batch: List[jnp.array],
        tanh_constant: float
    ) -> jnp.array:
        def tanh_vectorize(img: jnp.array, tanh_constant=tanh_constant):
            img = img / 255.0
            img = img - 0.5
            img = img * tanh_constant

            arctanh_img = jnp.arctanh(img)

            return arctanh_img

        return jax.vamp(tanh_vectorize)(batch)

    @staticmethod
    @jax.jit
    def reverse_tanh_image_batch(
        batch: List[jnp.array],
        tanh_constant: float,
    ) -> jnp.array:
        def reverse_tanh(img):
            return jnp.tanh(img) / (tanh_constant + 0.5) * 255

        return jax.vmap(reverse_tanh)(batch)

    @staticmethod
    @jax.jit
    def clip_img_batch(
        batch: jnp.array,
        max_val: float,
        tanh_constant: float
    ) -> jnp.array:
        def clip(img, max_val=max_val):
            img = jnp.clip(img, 0, max_val)

            return img

        reversed_imgs = ImageHandler.reverse_tanh_image_batch(
            batch,
            tanh_constant
        )

        return jax.vamp(clip)(reversed_imgs)

    @staticmethod
    def resize_tensor(input_tensor, model_input_shape):
        if input_tensor.shape[1:] == model_input_shape or model_input_shape[1] is None:
            return input_tensor
        resized_tensor = jax.image.resize(input_tensor, model_input_shape[:2])
        return resized_tensor
