import os
import requests
from tqdm import tqdm
import zipfile

link_list = {
    "arcface.h5": "https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5",
    "deepid.h5": "https://github.com/serengil/deepface_models/releases/download/v1.0/deepid_keras_weights.h5",
    "facenet.h5": "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5",
    "facenet512.h5": "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5",
    "deepface.h5.zip": "https://github.com/swghosh/DeepFace/releases/download/weights-vggface2-2d-aligned/VGGFace2_DeepFace_weights_val-0.9034.h5.zip",
    "openface.h5": "https://github.com/serengil/deepface_models/releases/download/v1.0/openface_weights.h5",
    "vggface.h5": "https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5",
}

MODEL_PATH = "face_models"

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

print("Downloading models...")

pbar = tqdm(total=len(link_list))

for name, url in link_list.items():
    print(f"Dowloading {name}")
    
    path = os.path.join(MODEL_PATH, name)

    if not os.path.exists(path):   
        resp = requests.get(url)

        with open(path, "wb") as fwb:
            fwb.write(resp.content)
    
    print(f"{name} successfully downloaded at {path}")
    
    if name.split(".")[-1] == "zip":
        with zipfile.ZipFile(path) as zf:
            zf.extractall()
            
        os.rename("VGGFace2_DeepFace_weights_val-0.9034.h5", "deepface.h5")
    
    pbar.update(1)

print("Done downloading models.")
pbar.close()