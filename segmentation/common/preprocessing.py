from os import listdir
from os.path import join
import matplotlib.pyplot as plt
from numpy import array, asarray, loadtxt
from PIL import Image

CROP = (221, 120, 791, 719)
SIZE = (1200, 850)


def crop_image(img: Image) -> Image:
    """Crops an image if its size matches ones of Prague dataset images.

    Parameters
    ----------
    img : Image
        Image to be processed.

    Returns
    -------
    Image
        Processed image.
    """
    if img.size == SIZE:
        pass #img = img.crop(CROP)

    return img


def load_position(dir_path: str, label_file: str) -> array:
    """Loads position of an object from a file.

    Parameters
    ----------
    dir_path : str
        Folder of the file.
    label_file : str
        File name.

    Returns
    -------
    array
        Position of an object defined by a bounding box.
    """
    label = loadtxt(join(dir_path, label_file), delimiter=";")

    transformed_label = asarray(
        [
            label[0] - label[2] / 2,
            label[1] - label[3] / 2,
            label[0] + label[2] / 2,
            label[1] + label[3] / 2,
        ]
    )

    return transformed_label


def load_imgs_dir(dir_path: str) -> list:
    """Loads images from a directory.

    Parameters
    ----------
    dir_path : str
        Folder containing images to load.

    Returns
    -------
    list
        List of PIL Images.
    """
    data = []
    for file in sorted(listdir(dir_path)):
        file_path = join(dir_path, file)
        img = Image.open(file_path)
        data.append(img)

    return data


def load_img(dir_path: str, img_file: str, crop: bool = True) -> Image:
    """Loads an images.

    Parameters
    ----------
    dir_path : str
        Folder containing images to load.
    img_file : str
        File name.
    crop : bool
        Flag describing if an image should be cropped.

    Returns
    -------
    Image
        Loaded Image.
    """
    joined_path = join(dir_path, img_file)
    img = Image.open(joined_path)
    img = img.convert('RGB')
    if crop:
        img = crop_image(img)
    return img
