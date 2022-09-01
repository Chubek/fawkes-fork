import json
import os
from glob import glob
from typing import List

import jsonpickle
import numpy as np
from dotenv import dotenv_values
from fawkes.fork.face_op import FaceBase
from scipy.spatial.distance import euclidean

CONFIG = dotenv_values(".env")

TARGET_PICKLE_PATH = CONFIG['TARGET_PICKLE_PATH']
PICKLED_KMEANS_PATH = CONFIG['PICKLED_KMEANS_PATH']
SELECTED_MODEL = CONFIG['MODEL_TO_USE']


def find_furthest_cluster(
    source_imgs: List[FaceBase]
) -> List[FaceBase]:
    with open(PICKLED_KMEANS_PATH, "r") as fr:
        content = fr.read()
        kmeans = jsonpickle.decode(content)

    files = glob(f"{TARGET_PICKLE_PATH}/*.json")

    target_imgs = []

    def get_furthest(
        i,
        source_imgs=source_imgs,
        files=files
    ):
        simg = source_imgs[i]

        feat = simg.feature_repr[SELECTED_MODEL]
        feat = [i, *feat.flatten().tolist()]

        centrd = kmeans.predict(feat)[0]
        cc = kmeans.cluster_centers_[centrd]
        dists = [(i, euclidean(cc, oc))
                 for i, oc in
                 enumerate(kmeans.cluster_centers_)]

        max_dist = max(dists, key=lambda x: x[1])[0]
        furthest_label = [i for i, l in enumerate(
            kmeans.labels_) if l == max_dist]
        random_choice = np.random.choice(furthest_label)
        f = files[random_choice]

        with open(f, "r") as fr:
            content = fr.read()
            obj = jsonpickle.decode(content)

            target_imgs.append(obj)

    np.vectorize(get_furthest)(np.arange(0, len(source_imgs)))

    return target_imgs
