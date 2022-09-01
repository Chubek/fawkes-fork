from __future__ import annotations

from typing import Any, Dict, List, Tuple
from glob import glob


import cv2
import jax
import jax.numpy as jnp
from mtcnn import MTCNN
from pydantic import BaseModel

from .image_models import ImageModelOps

detector = MTCNN()


class FaceBase(BaseModel):
    img_path: str
    resize: Tuple[int, int] = (224, 224)
    img_data: jnp.array = jnp.asarray([])
    face_cropped: jnp.array = jnp.asarray([])
    face_cropped_tanh: jnp.array = jnp.asarray([])
    feature_repr: Dict = {}
    box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    protected_face: jnp.arrray = jnp.array([])

    @classmethod
    def load_and_new(cls, img_path: str, resize=(224, 224)):
        obj = cls(img_path=img_path, resiz=resize)

        obj.load_image()
        obj.detect_face()
        obj.tanh_face()
        obj.feat_repr()

        return obj

    def load_image(
        self
    ):
        self.img_data = cv2.imread(self.img_path)
        self.img_data = cv2.cvtColor(self.img_data, cv2.COLOR_BGR2RGB)
        self.img_data = self.img_data.astype(jnp.int8)

    def detect_face(
        self
    ):
        detected_faces = detector.detect_faces(self.img_data)

        if len(detected_faces) == 0:
            raise ValueError("No faces detected")

        x, y, w, h = detected_faces[0]['box']

        self.detect_face = self.img_data[y:y + h, x:x + w, :]
        self.detect_face = cv2.resize(self.detect_face, self.resize)
        self.box = (x, y, w, h)

    def tanh_face(
        self
    ):
        self.face_cropped_tanh = jnp.tanh(self.face_cropped)

    def feat_repr(
        self
    ):
        self.feature_repr = ImageModelOps.load_feature_reprt(
            self.face_cropped_tanh)

    def reassemble(
        self
    ) -> jnp.array:
        img_data_copy = self.img_data.copy()

        x, y, w, h = self.box

        img_data_copy[img_data_copy[y:y + h, x:x + w, 3]] = self.protected_face

