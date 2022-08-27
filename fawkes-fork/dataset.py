from __future__ import annotations
from ctypes import Union
from tempfile import tempdir

import jax.numpy as jnp
import jax

from pydantic import BaseModel
from enum import Enum

import os

from typing import List, Optional

from .facial_detection import FacialDetector
from .feature_extraction import FeatureExtractor
from .utils import Utils

import cv2


class ImageType(Enum):
    Init = "init"
    Source = "source"
    Target = "target"
    Modded = "modded"

class SingleImage(BaseModel):
    img_path: str = ""
    img_data: jnp.array = jnp.asarray([])
    img_type: ImageType = ImageType.Init
    detected_faces: Optional[jnp.array] = None
    feature_space: Optional[jnp.array] = None
    merged_faces: Optional[jnp.array] = None

    @classmethod
    def new_source(cls, img_path: str) -> SingleImage:
        return cls(
            img_path=img_path,
            image_type=ImageType.Source 
        )

    @classmethod
    def new_target(cls, img_path: str) -> SingleImage:
        return cls(
            img_path=img_path,
            image_type=ImageType.Target 
        )

    @classmethod
    def new_modded(cls, img_data: jnp.array, tmp_folder: str) -> SingleImage:
        rand_name = Utils.random_str()
        file_name = rand_name + ".png"
        
        img_path = os.path.join(tmp_folder, file_name)

        
        return cls(
            img_path=img_path,
            img_data=img_data,
            image_type=ImageType.Modded 
        )



class ImageDataSet:
    def __init__(self) -> None:
        self.list = []


    @classmethod
    def source_from_list(cls, img_paths: List[str]) -> ImageDataSet:
        list = [SingleImage.new_source(i) for i in img_paths]

        obj = cls()

        obj.list = list

        return obj

    @classmethod
    def target_from_list(cls, img_paths: List[str]) -> ImageDataSet:
        list = [SingleImage.new_target(i) for i in img_paths]

        obj = cls()

        obj.list = list

        return obj

    @classmethod
    def modded_from_arrays(cls, img_data: List[jnp.array]) -> ImageDataSet:
        list = [SingleImage.new_modded(a) for a in img_data]

        obj = cls()

        obj.list = list

        return obj


    def __getitem__(self, idx: int) -> SingleImage:
        return self.list[idx]


    def __setitem__(self, idx: int, data: SingleImage):
        self.list[idx] = data




