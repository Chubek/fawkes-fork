from fawkes.fork.face_op import FaceBase
from tqdm import tqdm
from glob import glob
import jsonpickle
import os

TARGET_IMGS = "target_data"

def load_and_serialize_target_images():
    targets = glob(f"{TARGET_IMGS}/*.[jp][pn][g]")

    list_imgs = []
    
    pbar = tqdm(total=len(targets))

    for tr in targets:
        fb = FaceBase.load_and_new(tr)

        list_imgs.append(fb)

        pbar.update(1)

    pbar.close()

    pickled = jsonpickle.encode(list_imgs)

    with open(os.path.join(TARGET_IMGS, "pickled_obj.json", "w")) as fr:
        fr.write(pickled)


if __name__ == "__main__":
    load_and_serialize_target_images()