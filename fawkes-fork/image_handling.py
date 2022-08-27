from typing import List, Tuple
import cv2
import os
import jax.numpy as jnp
import jax
import glob
from pydantic import BaseModel


class ImageHandler(BaseModel):
    source_image_path: str
    target_image_path: str
    tanah_constant: float
    max_val: float

    @jax.jit
    def load_and_parse_img(self) -> Tuple[
        jnp.array[jnp.array],
        jnp.array[jnp.array]
    ]:
        def load_single_image(path: str) -> jnp.array:
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_jax = jnp.asarray(img_rgb)

            return img_jax

        def get_imgs_path(path: str) -> List[str]:
            paths = glob.glob(os.path.join(path, "/*/*.[jp][pn][g]"))

            return paths

        found_source = get_imgs_path(self.source_images_path)
        found_target = get_imgs_path(self.target_images_path)

        loaded_source = jax.vmap(load_single_image)(found_source)
        loaded_target = jax.vmap(load_single_image)(found_target)

        (loaded_source, loaded_target)

    @jax.jit
    def tanh_image_batch(self, batch: jnp.array) -> jnp.array:
        def tanh_vectorize(img: jnp.array, tanh_constant=self.tanh_constant):
            img = img / 255.0
            img = img - 0.5
            img = img * tanh_constant

            arctanh_img = jnp.arctanh(img)

            return arctanh_img

        return jax.vamp(tanh_vectorize)(batch)

    @jax.jit
    def reverse_tanh_image_batch(self, batch: jnp.array) -> jnp.array:
        def reverse_tanh(img): return jnp.tanh(img) / \
            (self.tanh_constant + 0.5) * 255

        return jax.vmap(reverse_tanh)(batch)

    @jax.jit
    def clip_img_batch(self, batch: jnp.array) -> jnp.array:
        def clip(img, max_val=self.max_val):
            img = jnp.clip(img, 0, max_val)

            return img

        reversed_imgs = self.reverse_tanh_image_batch(
            batch, self.tanh_constant)

        return jax.vamp(clip)(reversed_imgs)


    @staticmethod
    def resize_tensor(input_tensor, model_input_shape):
        if input_tensor.shape[1:] == model_input_shape or model_input_shape[1] is None:
            return input_tensor
        resized_tensor = jax.image.resize(input_tensor, model_input_shape[:2])
        return resized_tensor