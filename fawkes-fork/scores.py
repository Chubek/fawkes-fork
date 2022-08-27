from typing import Tuple
import jax.numpy as jnp
import jax.scipy
import jax
from dm_pix import ssim


class ScoreTerpentine:
    @staticmethod
    @jax.jit
    def dissim_score(
        source_raw_batch: jnp.array,
        source_mod_raw_batch: jnp.array,
        l_threshold: jnp.array
    ) -> Tuple[jnp.array, jnp.array, float, float]:

        def ssim_single(pair: Tuple[jnp.array, jnp.array]):
            img_a, img_b = pair
            return ssim(img_a, img_b)

        ssim_batch = jax.vmap(
            ssim_single)([
                (a, b)
                for a, b
                in zip(
                    source_raw_batch,
                    source_mod_raw_batch,
                )])

        dist_raw = (1.0 - jnp.stack(ssim_batch)) / 2.0
        dist = jnp.maximum(dist_raw - l_threshold, 0.0)

        dist_raw_avg = jnp.mean(dist_raw)
        dist_sum = jnp.sum(dist)

        return (dist, dist_raw, dist_sum, dist_raw_avg)
