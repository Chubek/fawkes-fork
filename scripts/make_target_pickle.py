import os
from glob import glob

import jsonpickle
from dotenv import dotenv_values
from pysrc.fawkes.fork.face_op import FaceBase
from tqdm import tqdm

CONFIG = dotenv_values(".env")

TARGET_IMGS = CONFIG['TARGET_IMGS']
TARGET_PICKLE_PATH = CONFIG['TARGET_PICKLE_PATH']

if not os.path.exists(TARGET_PICKLE_PATH):
    os.makedirs(TARGET_PICKLE_PATH)


def load_and_serialize_target_images():
    targets = glob(f"{TARGET_IMGS}/*.[jp][pn][g]")

    pbar = tqdm(total=len(targets))

    for i, tr in enumerate(targets):
        try:
            fb = FaceBase.load_and_new(tr)
        except:
            pbar.update(1)
            continue

        pickled = jsonpickle.encode(fb)

        with open(os.path.join(TARGET_PICKLE_PATH, f"target_pickled_{i}.json"), "w") as fw:
            fw.write(pickled)

        pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    load_and_serialize_target_images()
