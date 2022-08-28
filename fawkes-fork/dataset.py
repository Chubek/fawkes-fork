from __future__ import annotations

import os
from ctypes import Union
from enum import Enum
from tempfile import tempdir
from tokenize import Single
from typing import List, Optional, Tuple

import cv2
import jax
import jax.numpy as jnp
from pydantic import BaseModel

from .facial_detection import DetectedFace, FacialDetector
from .feature_extraction import FeatureExtractor
from .image_handling import ImageHandler
from .utils import Utils

from copy import deepcopy


class ImageType(Enum):
    Init = "init"
    Source = "source"
    Target = "target"
    Cropped = "cropped"
    Modded = "modded"
    Arctanned = "Arctanned"
    RevArctanned = "RevArctanned"
    Clipped = "Clipped"
    WithoutFace = "WithoutFace"


class SingleImage(BaseModel):
    img_path: str = ""
    img_data: jnp.array = jnp.asarray([])
    img_type: ImageType = ImageType.Init
    detected_face: Optional[DetectedFace] = None
    feature_space: Optional[jnp.array] = None
    merged_faces: Optional[jnp.array] = None
    arctanned: Optional[jnp.array] = None
    reverse_arctanned: Optional[jnp.array] = None
    wo_face: Optional[jnp.array] = None
    clipped: Optional[jnp.array] = None

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
        cv2.imwrite(img_path, img_data)

        return cls(
            img_path=img_path,
            img_data=img_data,
            image_type=ImageType.Modded
        )

    @classmethod
    def new_cropped(cls, img_data: jnp.array, tmp_folder: str) -> SingleImage:
        rand_name = Utils.random_str()
        file_name = rand_name + ".png"

        img_path = os.path.join(tmp_folder, file_name)
        cv2.imwrite(img_path, img_data)

        return cls(
            img_path=img_path,
            img_data=img_data,
            image_type=ImageType.Cropped
        )

    @classmethod
    def new_arctanned(cls, img_data: jnp.array, tmp_folder: str) -> SingleImage:
        rand_name = Utils.random_str()
        file_name = rand_name + ".png"

        img_path = os.path.join(tmp_folder, file_name)
        cv2.imwrite(img_path, img_data)

        return cls(
            img_path=img_path,
            img_data=img_data,
            image_type=ImageType.Arctanned
        )

    @classmethod
    def new_clipped(cls, img_data: jnp.array, tmp_folder: str) -> SingleImage:
        rand_name = Utils.random_str()
        file_name = rand_name + ".png"

        img_path = os.path.join(tmp_folder, file_name)
        cv2.imwrite(img_path, img_data)

        return cls(
            img_path=img_path,
            img_data=img_data,
            image_type=ImageType.Clipped
        )

    @classmethod
    def new_rev_arctanned(cls, img_data: jnp.array, tmp_folder: str) -> SingleImage:
        rand_name = Utils.random_str()
        file_name = rand_name + ".png"

        img_path = os.path.join(tmp_folder, file_name)
        cv2.imwrite(img_path, img_data)

        return cls(
            img_path=img_path,
            img_data=img_data,
            image_type=ImageType.RevArctanned
        )

    @classmethod
    def new_wo_face(cls, img_data: jnp.array, tmp_folder: str) -> SingleImage:
        rand_name = Utils.random_str()
        file_name = rand_name + ".png"

        img_path = os.path.join(tmp_folder, file_name)
        cv2.imwrite(img_path, img_data)

        return cls(
            img_path=img_path,
            img_data=img_data,
            image_type=ImageType.WithoutFace
        )

    @staticmethod
    @jax.jit
    def run_batch_load(
        ls: List[SingleImage],
        resize: Tuple[int, int]
    ) -> List[SingleImage]:
        paths = [si.img_path for si in ls]

        images_loaded = ImageHandler.load_image_batch(paths, resize=resize)

        fin = []

        for ls1, ls2 in zip(ls, images_loaded):
            ls1.img_data = ls2

            fin.append(ls1)

        return fin

    @staticmethod
    @jax.jit
    def run_arctan_from_array(
        ls: List[SingleImage],
        tanh_constant: float
    ) -> List[SingleImage]:
        single_imgs_loaded = SingleImage.run_batch_load_source(ls)
        ls_arrays = [si.img_array for si in single_imgs_loaded]

        loaded_arctans = ImageHandler.tanh_image_batch(
            ls_arrays, tanh_constant)

        fin = []

        for ls1, ls2 in zip(ls, loaded_arctans):
            ls1.arctanned = ls2

            fin.append(ls1)

        return fin

    @staticmethod
    @jax.jit
    def run_revarctan_from_array(
        ls: List[SingleImage],
        tanh_constant: float
    ) -> List[SingleImage]:
        single_imgs_loaded = SingleImage.run_batch_load_source(ls)
        ls_arrays = [si.img_array for si in single_imgs_loaded]

        loaded_arctans = ImageHandler.reverse_tanh_image_batch(
            ls_arrays, tanh_constant)

        fin = []

        for ls1, ls2 in zip(ls, loaded_arctans):
            ls1.reverse_arctanned = ls2

            fin.append(ls1)

        return fin

    @staticmethod
    @jax.jit
    def run_clipped_from_array(
        ls: List[SingleImage],
        max_val: float,
        tanh_constant: float
    ) -> List[SingleImage]:
        single_imgs_loaded = SingleImage.run_batch_load_source(ls)
        ls_arrays = [si.img_array for si in single_imgs_loaded]

        loaded_arctans = ImageHandler.clip_img_batch(
            ls_arrays, max_val, tanh_constant)

        fin = []

        for ls1, ls2 in zip(ls, loaded_arctans):
            ls1.clipped = ls2

            fin.append(ls1)

        return fin

    @staticmethod
    @jax.jit
    def run_batch_detect(
        ls: List[SingleImage],
        detection_method: FacialDetector,
        target_size: Tuple[int, int]
    ) -> List[SingleImage]:
        img_paths = [i.img_path for i in ls]

        detected_faces = FacialDetector.detect_face_batch(
            img_paths,
            detection_method,
            target_size
        )

        fin = []

        for ls1, ls2 in zip(ls, detected_faces):
            ls1.detected_face = ls2

            fin.append(ls1)

        return fin

    @staticmethod
    @jax.jit
    def run_extract_features(
        ls: List[SingleImage],
        extraction_method: FeatureExtractor,
        detection_method: FacialDetector
    ) -> List[SingleImage]:
        img_paths = [i.img_path for i in ls]

        feature_extracted = FeatureExtractor.extract_feature_batch(
            img_paths,
            extraction_method,
            detection_method
        )

        fin = []

        for ls1, ls2 in zip(ls, feature_extracted):
            ls1.feature_space = ls2

            fin.append(ls1)

        return fin

    @staticmethod
    @jax.jit
    def run_remove_face(
        ls: List[SingleImage],
        extraction_method: FeatureExtractor,
        detection_method: FacialDetector
    ) -> List[SingleImage]:
        tuples = [(i.img_array, i.detected_face) for i in ls]

        imgs_wo_face = DetectedFace.get_wo_face_batch(
            tuples
        )

        fin = []

        for ls1, ls2 in zip(ls, imgs_wo_face):
            ls1.wo_face = ls2

            fin.append(ls1)

        return fin

    @staticmethod
    def run_apply_pixel_to_faces(
        ls:  List[SingleImage]
    ):
        pass

    def delete_image_on_server(self):
        os.remove(self.img_path)


class ImageDataSet:
    def __init__(self) -> None:
        self.list = []
        self.type = ImageType.Init
        self.cur_index = 0

    @classmethod
    def source_from_list(
        cls,
        img_paths: List[str],
        resize: Tuple[int, int]
    ) -> ImageDataSet:
        list = [SingleImage.new_source(i) for i in img_paths]
        list = SingleImage.run_batch_load(list, resize)

        obj = cls()

        obj.list = list
        obj.type = ImageType.Source

        return obj

    @classmethod
    def target_from_list(
        cls,
        img_paths: List[str],
        resize: Tuple[int, int]
    ) -> ImageDataSet:
        list = [SingleImage.new_target(i) for i in img_paths]
        list = SingleImage.run_batch_load(list, resize)

        obj = cls()

        obj.list = list
        obj.type = ImageType.Target

        return obj

    @classmethod
    def modded_from_arrays(cls, img_data: List[jnp.array]) -> ImageDataSet:
        list = [SingleImage.new_modded(a) for a in img_data]
        list = SingleImage.run_apply_pixel_to_faces(list)

        obj = cls()

        obj.list = list
        obj.type = ImageType.Modded

        return obj

    @classmethod
    def cropped_from_arrays(
        cls,
        img_data: List[jnp.array],
        detection_method: FacialDetector,
        target_size: Tuple[int, int]
    ) -> ImageDataSet:
        list = [SingleImage.new_cropped(cr) for cr in img_data]
        list = SingleImage.run_batch_detect(
            list, detection_method, target_size)

        obj = cls()

        obj.list = list
        obj.type = ImageType.Cropped

        return obj

    @classmethod
    def arctanned_from_arrays(
        cls,
        img_data: List[jnp.array],
        tanh_constant: float
    ) -> ImageDataSet:
        list = [SingleImage.new_arctanned(cr) for cr in img_data]
        list = SingleImage.run_arctan_from_array(list, tanh_constant)

        obj = cls()

        obj.list = list
        obj.type = ImageType.Arctanned

        return obj

    @classmethod
    def clipped_from_arrays(
        cls,
        img_data: List[jnp.array],
        max_val: float,
        tanh_constant: float
    ) -> ImageDataSet:
        list = [SingleImage.new_clipped(cr) for cr in img_data]
        list = SingleImage.run_clipped_from_array(
            list,
            max_val,
            tanh_constant
        )

        obj = cls()

        obj.list = list
        obj.type = ImageType.Clipped

        return obj

    @classmethod
    def rev_arctanned_from_arrays(
        cls,
        img_data: List[jnp.array],
        tanh_constant: float
    ) -> ImageDataSet:
        list = [SingleImage.new_rev_arctanned(cr) for cr in img_data]
        list = SingleImage.run_revarctan_from_array(list, tanh_constant)

        obj = cls()

        obj.list = list
        obj.type = ImageType.RevArctanned

        return obj

    @classmethod
    def wo_face_from_arrays(
        cls,
        img_data: List[jnp.array],
        extraction_method: FeatureExtractor,
        detection_method: FacialDetector
    ) -> ImageDataSet:
        list = [SingleImage.new_wo_face(cr) for cr in img_data]
        list = SingleImage.run_remove_face(
            list, extraction_method, detection_method)

        obj = cls()

        obj.list = list
        obj.type = ImageType.WithoutFace

        return obj

    def __getitem__(self, idx: int) -> SingleImage:
        return self.list[idx]

    def __setitem__(self, idx: int, data: SingleImage):
        self.list[idx] = data

    def __add__(self, other: ImageDataSet) -> ImageDataSet:
        ls = deepcopy(self.list)
        ls_other = deepcopy(other.list)

        sum_ls = ls.extend(ls_other)

        obj = deepcopy(self)

        obj.list = sum_ls

        return obj

    def __iter__(self) -> ImageDataSet:
        self.cur_index = 0
        return self

    def __next__(self) -> SingleImage:
        if self.cur_index <= len(self.list):
            return self.list[self.cur_index]
        else:
            raise StopIteration
