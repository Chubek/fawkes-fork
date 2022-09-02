from .facenet import InceptionResNetV2
import os

MODEL_PATH = "face_models"

def loadFaceNet512():

    model = InceptionResNetV2(dimension = 512)
    model.load_weights(os.path.join(MODEL_PATH, "facenet512.h5"))

    return model