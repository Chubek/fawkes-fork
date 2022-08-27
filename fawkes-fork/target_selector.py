from __future__ import annotations
from dis import dis

from scipy.spatial.distance import cdist
from enum import Enum
from time import time
import jax
import jax.numpy as jnp

from pydantic import BaseModel

class DistanceType(Enum):
    Euclidean = "euclidean"
    CityBlock = "cityblock"
    Cosine = "cosine"
    Jaccard = "jaccard"


    @staticmethod
    def from_str(s: str) -> DistanceType:
        s = s.lower()

        if s == "euclidean" or s == "l2":
            return DistanceType.Euclidean
        elif s == "cityblock" or s == "manhattan":
            return DistanceType.CityBlock
        elif s == "cosine":
            return DistanceType.Cosine
        elif s == "jaccard":
            return DistanceType.Jaccard

    @staticmethod
    def to_str(dtype: DistanceType) -> str:
        if dtype == DistanceType.Euclidean:
            return "euclidean"
        elif dtype == DistanceType.CityBlock:
            return "cityblock"
        elif dtype == DistanceType.Cosine:
            return "cosine"
        elif dtype == DistanceType.Jaccard:
            return "jaccard"
    

    @staticmethod
    @jax.jit
    def calculate_distancne(
        array_a: jnp.array,
        array_b: jnp.array,
        distance_type: DistanceType
    ) -> jnp.array:
        dist_type = DistanceType.to_str(distance_type)

        func_dist = lambda a, b: cdist(a, b, dist_type)

        return jax.vmap(func_dist)(array_a, array_b)


class GammaRand:
    @staticmethod
    def generate_random_gamma(c: int, len_df: int):
        key = jax.random.PRNGKey(time())

        a = jax.random.uniform(1e-5, 1.0, [c, len_df])

        return jax.random.gamma(key, a, [c, len_df])


class FuzzyCMeans(BaseModel):
    df: jnp.array = jnp.asarray([])
    len_df: int = 100
    c: int = 3
    m: float = 3.0
    distance_type: DistanceType = DistanceType.Euclidean
    tolerance: float = jax.random.uniform(jax.random.PRNGKey(time()))
    gamma: jnp.array = jnp.asarray([]) 
    gamma_prev: jnp.array = jnp.asarray([]) 
    centroids: jnp.array = jnp.asarray([])


    @classmethod
    def init_new(
        cls,
        data_frame: jnp.array,
        c: int,
        m: float,
        distance_type: DistanceType
    ) -> FuzzyCMeans:
        obj = cls()

        obj.df = data_frame
        obj.len_df = data_frame.shape[0]
        obj.c = c
        obj.m = m
        obj.distance_type = distance_type
        obj.gamma = GammaRand.generate_random_gamma(c, data_frame.shape[0])
        obj.gamma_prev = GammaRand.generate_random_gamma(c, data_frame.shape[0])

        obj.initialize_calc_centroids()

        return obj

    def initialize_calc_centroids(self):    
        gamma_ = self.gamma * self.m
        cents = []

        def cent_calc(gamma_arg: jnp.array, df=self.df, gamma_=gamma_):
            sums = []
            calc = lambda x: sums.append(jnp.dot(gamma_arg, x))

            jax.vmap(calc)(df)

            sum_of_sum = jnp.sum(jnp.asarray(sums), axis=-1)
            sum_div = sum_of_sum / jnp.sum(gamma_)

            cents.append(sum_div)


        jax.vmap(cent_calc)(gamma_)

        self.centroids = jnp.asarray(cents)

    
    def check_convergence(self) -> bool:
        return jnp.sum((self.gamma - self.gamma_prev) ** 2) <= 2

    def get_distance(self, a: jnp.array, b: jnp.array) -> jnp.array:
        distance_type = self.distance_type.to_str()

        return cdist(a, b, metric=distance_type)