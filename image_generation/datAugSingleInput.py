import random
import numpy as np
import albumentations as A
import copy

"""
THIs code is taken from the https://gitlab.fel.cvut.cz/morozart/bachelor-thesis/-/blob/main/datAugSingleInput.py
"""


def Grid_distortion(image):
    """
    Function implements grid distortion transformation applied on input image
    :param image:
    :type ndarray
    :return: grid_dist_img:
    :type ndarray
    """
    transform = A.Compose(
        [
            A.GridDistortion(
                num_steps=10,
                distort_limit=0.25,
                interpolation=1,
                border_mode=0,
                always_apply=True,
                p=1,
            ),
        ]
    )
    grid_dist_img = transform(image=image)["image"]
    return grid_dist_img


def Elastic_transform(image):
    """
    Function implements elastic transformation applied on input image
    :param image:
    :type ndarray
    :return: el_trans_img:
    :type ndarray
    """
    transform = A.Compose(
        [
            A.ElasticTransform(
                alpha=10,
                sigma=50,
                alpha_affine=15,
                interpolation=1,
                border_mode=0,
                always_apply=True,
                p=1,
            )
        ]
    )
    el_trans_img = transform(image=image)["image"]
    return el_trans_img
