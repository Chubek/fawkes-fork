import os
from glob import glob
from typing import Any, Iterable

import jsonpickle
import numpy as np
from dotenv import dotenv_values
from sklearn.cluster import KMeans

CONFIG = dotenv_values(".env")

TARGET_PICKLE_PATH = CONFIG['TARGET_PICKLE_PATH']
NUM_CLUSTERS = CONFIG['NUM_CLUSTERS']
PICKLED_KMEANS_PATH = CONFIG['PICKLED_KMEANS_PATH']
MODEL_TO_USE = CONFIG['MODEL_TO_USE']


ALL_JSONS = glob(f"{TARGET_PICKLE_PATH}/*.json")


if not os.path.exists(PICKLED_KMEANS_PATH):
    os.makedirs(PICKLED_KMEANS_PATH)


def load_targets_and_get_feats() -> Iterable:
    list_feat = []

    def load_single(num):
        file = ALL_JSONS[num]

        if os.path.exists(file):
            with open(file, "r") as fr:
                content = fr.read()
                obj = jsonpickle.decode(content)
                obj_feat = obj.feature_repr[MODEL_TO_USE]
                list_feat.append([num, *obj_feat.flatten().tolist()])

    range = np.arange(0, len(ALL_JSONS))

    np.vectorize(load_single)(range)

    return np.asarray(list_feat, dtype=object)


def calculate_clusters(list_feat: Iterable) -> Any:
    kmeans = KMeans(n_clusters=NUM_CLUSTERS)
    kmeans.fit(list_feat)

    return kmeans


def main():
    list_feat = load_targets_and_get_feats()
    kmeans = calculate_clusters(list_feat)

    pickled_kmeans = jsonpickle.encode(kmeans)

    with open(os.path.join(PICKLED_KMEANS_PATH, "pickled_kmeans.json"), "w") as fw:
        fw.write(pickled_kmeans)


if __name__ == "__main__":
    main()
