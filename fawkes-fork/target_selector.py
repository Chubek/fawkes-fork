from __future__ import annotations
from dis import dis
from typing import List, Optional, Tuple

from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean

import jax
import jax.numpy as jnp

from pydantic import BaseModel


class TargetSelector(BaseModel):
    k: int = 3
    radius: float = 1.5
    selected_targets: List[jnp.array] = []
    image_array_list: jnp.array = jnp.asarray([])
    kd_tree: Optional[KDTree] = None
    k_neighbors: jnp.array = jnp.asarray([])
    k_distances: jnp.array = jnp.asarray([])
    local_reachable_density: jnp.asarray = jnp.array([])
    local_outlier_factor: jnp.asarray = jnp.array([])


    @classmethod
    def init_and_compute(cls, image_array_list: jnp.array, k: int, radius: float) -> List[jnp.array]:
        obj = cls()

        obj.k = k
        obj.radius = radius
        obj.image_array_list = image_array_list

        selected_targets = obj.compute()

        return obj.compute()

    def create_kdtree(self):
        self.kd_tree = KDTree(self.image_array_list)

    @jax.jit
    def compute(self) -> List[jnp.array]:
        self.calculate_knn()
        self.calculate_k_distance()
        self.calculate_lrd()
        self.calculate_fob()
        
        self.sort_and_get_top()

        return self.selected_targets
          

    @jax.jit
    def calculate_knn(self):
        knn = []

        def get_knn(a: jnp.array, k=self.k):
            knn_query = self.kd_tree.query(a, k)

            knn.append((knn, knn_query))

        jax.vmap(get_knn)(self.image_array_list)

        self.k_neighbors = jnp.asarray(knn)


    @jax.jit
    def calculate_k_distance(self):
        k_dist = []

        def calc_k_dist(a: jnp.array):
            p = a[0]
            knn = a[1]

            dist_list = []

            def dist(a: jnp.array, b: jnp.array):
                distance = euclidean(a, b)

                dist_list.append((a, b, distance))

            jax.vmap(dist)(p, knn)

            k_dist.append(jnp.asarray(dist))

        
        jax.vmap(calc_k_dist)(self.k_neighbors)


        self.k_distances = jnp.asarray(k_dist)


    @jax.jit
    def calculate_lrd(self):
        lrds = []        

        def get_lrd(k_dist_pairs: jnp.array):
            rds = []
            
            def get_rd(k_dist_pair: jnp.array, k=self.k) -> jnp.array:
                _, _, distance = k_dist_pair
                
                rds.append(jnp.max(k - distance, distance))


            jax.vmap(get_rd)(k_dist_pairs)

            a, _, _ = k_dist_pairs
            
            avg_rds_inversed = 1.0 / (jnp.mean(rds))
            lrds.append((a, avg_rds_inversed))

                   

        jax.vmap(get_lrd)(self.k_distances)

        self.local_reachable_density = jnp.asarray(lrds)


    @jax.dist
    def calculate_fob(self):
        fobs = []
        mean_lrds = jnp.mean([lrd[1] for lrd in self.local_reachable_density])
        
        def calc_fob(lrd_arg, mean_lrds=mean_lrds):
            a, lrd = lrd_arg

            fobs.append((a, mean_lrds * (1.0 / lrd)))


        jax.vmap(calc_fob)(self.local_reachable_density)

        self.local_outlier_factor = jnp.array(fobs)
         


    @jax.jit
    def sort_and_get_top(self):
        sort = sorted(self.local_outlier_factor, key=lambda x:x[1])

        self.selected_targets = [a for a in sorted[:len(sorted) // 4]]