from __future__ import annotations

from typing import Any, Dict, List, Tuple, Iterable
from glob import glob
import base64 as bs

import cv2
import jax
import jax.numpy as jnp
from mtcnn import MTCNN
from pydantic import BaseModel
import numpy as np
from .image_models import ImageModelOps

detector = MTCNN()


class FaceBase(BaseModel):
    img_path: str
    resize: Tuple[int, int] = (224, 224)
    img_data: Iterable = jnp.asarray([])
    face_cropped: Iterable = jnp.asarray([])
    face_cropped_tanh: Iterable = jnp.asarray([])
    feature_repr: Dict = {}
    box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    protected_face: Iterable = jnp.array([])
    final_img: Iterable = jnp.array([])

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
        #self.img_data = cv2.cvtColor(self.img_data, cv2.COLOR_BGR2RGB)

    def detect_face(
        self
    ):
        detected_faces = detector.detect_faces(self.img_data)

        if len(detected_faces) == 0:
            raise ValueError("No faces detected")

        x, y, w, h = detected_faces[0]['box']

        self.face_cropped = self.img_data[y:y + h, x:x + w, :]
        self.face_cropped = cv2.resize(self.face_cropped, self.resize)
        self.face_cropped = jnp.asarray(self.face_cropped)
        self.face_cropped = self.face_cropped.astype(jnp.int8)
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
        protected_face_copy = self.protected_face.copy()

        x, y, w, h = self.box
        i, j, _ = protected_face_copy.shape

        impeceble_pixels = protected_face_copy - self.face_cropped

        img_data_copy[y:y + i, x:x + j, :] += impeceble_pixels

        self.final_img = np.asarray(img_data_copy)

    def save_img(self, save_path: str):
        self.reassemble()

        cv2.imwrite(save_path, self.final_img)

    
    def img_to_b64_urlencoded(self) -> str:
        self.reassemble()
        
        _, buffer = cv2.imencode(".png", self.final_img)
        buffer_b64 = bs.b64encode(buffer)


        return f"data:image/png;base64, {buffer_b64}"