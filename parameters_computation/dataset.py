import torch
from torch import cat, int64, Tensor, unsqueeze, zeros, tensor
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose
from os.path import join
from PIL import Image
from os import listdir
import pandas as pd
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
)
import numpy as np
from transformations import SegCompose, SegCrop
from visualization import plot_image_label, plot_image

def label_to_mask(
    label: Tensor, plaque_with_wall: bool = False, encode_to_one_hot: bool = True
):
    """Processes label to a Tensor mask. If plaque_with_wall
    is selected to True, the plaque category is changed
    to the wall category. If encode_to_one_hot is selected to True,
    the mask is encoded in a one-hot setting.

    Parameters
    ----------
    label : Tensor
        Index of an item to return.
    plaque_with_wall : bool
        If True, the plaque and wall classes are united.
    encode_to_one_hot : bool
        If true, the reference is returned encoded in a one-hot setting,
        if false, the reference is returned with the classes encoded as an
        integer values.

    Returns
    -------
    Tensor
        Tensor label.
    """
    # print(label.shape)
    mask = cat((zeros(1, *label.shape[1:]), label))
    # print(mask.shape)
    mask = mask.argmax(0)
    # print(mask.shape)
    if plaque_with_wall:
        mask[mask == 2] = 1
        mask[mask == 3] = 2

    if encode_to_one_hot:
        mask = one_hot(mask).permute(2, 0, 1)
        mask[0, (label[0, ...] == 1) & (label[1, ...] == 1)] = 1

    return mask


TRANSFORMATIONS_TORCH = Compose(
    [
        Resize((512, 512)),
        ToTensor(),
    ]
)

def load_img(dir_path: str, img_file: str) -> Image:
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
    return img


def create_labels_df(path, echogenicity):
    label_dict = {}
    label_df = pd.read_csv(path)
    for i, row in label_df.iterrows():
        if pd.notna(row["echogenicity"]) and pd.notna(row["homogenicity"]):
            label_dict[row["img_name"]] = row["echogenicity"]-1 if echogenicity else row["homogenicity"]-1
    return label_dict

def create_regression_labels_dict(path):
    label_dict = {}
    label_df = pd.read_csv(path)
    for i, row in label_df.iterrows():
        if pd.notna(row["echogenicity"]) and pd.notna(row["homogenicity"]) and pd.notna(row["plaque_width"]):
            label_dict[row["img_name"]] = (row["plaque_width"],row["echogenicity"]-1, row["homogenicity"]-1)
    return label_dict


def img_and_plaque_mask(img, segmentation):
    return img, (segmentation == 2).to(torch.int).unsqueeze(0)
def img_and_one_hot_segmentation(img, segmentation):
    return img, one_hot(segmentation).permute(2, 0, 1)
def img_and_all_segments_mask(img, segmentation):
    return img, (segmentation != 0).to(torch.int).unsqueeze(0)

def mask_plaque(img, segmentation):
    img[:, segmentation == 0] = 0
    img[:, segmentation == 1] = 0
    img[:, segmentation == 3] = 0
    return img

def mask_all_segments(img, segmentation):
    img[:, segmentation == 0] = 0
    return img

def cat_segmentation_with_image(img, segmentation):
    return cat((img, one_hot(segmentation).permute(2, 0, 1)))

class RegressionDataset(Dataset):

    def __init__(
        self,
        images_path: str,
        segmentations_path: str,
        labels_path: str,
        masking_function,
        transformations_torch: Compose,
        transformations_custom: SegCompose,
        
    ) -> None:
        self.masking_function = masking_function
        self.images_path = images_path
        self.labels_path = labels_path
        self.segmentations_path = segmentations_path

        self.labels_dict = create_regression_labels_dict(self.labels_path)

        self.img_files = sorted(listdir(images_path))
        self.segmentation_files = sorted(listdir(segmentations_path))
        self.complete_images = []
        for segmentation_file in self.segmentation_files:
            if segmentation_file in self.img_files and segmentation_file in self.labels_dict:
                self.complete_images.append(segmentation_file)
        self.transformations_custom = transformations_custom
        self.transformations_torch = transformations_torch
        # self.transformations_torch_label = transformations_torch_label

    def __getitem__(self, index: int) -> tuple:
        """Gets item from the dataset at a specified index.

        Parameters
        ----------
        index : int
            Index of an item to return.

        Returns
        -------
        tuple
            Returns the processed image and the transformed segmentation mask
        """
        img = load_img(self.images_path, self.complete_images[index])
        segmentation = load_img(self.segmentations_path, self.complete_images[index])
        label = self.labels_dict[self.complete_images[index]]
        img, segmentation = self.transformations_custom(img, segmentation)
        img_tensor = self.transformations_torch(img)
        
        old_size = img.width, img.height
        # print(old_size)
        segmentation_tensor = self.transformations_torch(segmentation)
        # print(segmentation_tensor.shape)
        new_size = segmentation_tensor.shape[1], segmentation_tensor.shape[2]
        segmentation_mask = label_to_mask(segmentation_tensor, encode_to_one_hot=False)
        
        a = self.masking_function(img_tensor.clone(), segmentation_mask)
        # if label == 1:
        #     k = 0.9
        # elif label == 0:
        #     k = 0.1
        # result_tensor = torch.where(a != 0, k * torch.ones_like(a), a)
        # nonzero_indices = np.nonzero(a)
        # print(nonzero_indices)
        # # Get the non-zero values from the original tensor
        # # nonzero_values = a[nonzero_indices]
        # print(a[0, nonzero_indices[0][1], nonzero_indices[0][2]])
        # print(a[1, nonzero_indices[0][1], nonzero_indices[0][2]])
        # print(a[2, nonzero_indices[0][1], nonzero_indices[0][2]])
        # print(a.shape)
        # plot_image(a)
        # plot_image_label(img_tensor, segmentation_mask)
        # print(label[0])
        ### DELETE AFTER TESTING !!!!!
        # return result_tensor, tensor(label).long(), self.complete_images[index]
        ratio = (new_size[0] + new_size[1]) / (old_size[0] + old_size[1])
        label = label[0] * ratio
        # print(label)
        return a, tensor(label), self.complete_images[index]

    def __len__(self) -> int:
        """Returns a length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.complete_images)


class NonGeomParameterDataset(Dataset):

    def __init__(
        self,
        images_path: str,
        segmentations_path: str,
        labels_path: str,
        masking_function,
        echogenicity,
        transformations_torch: Compose,
        transformations_custom: SegCompose,
        
    ) -> None:
        self.masking_function = masking_function
        self.images_path = images_path
        self.labels_path = labels_path
        self.segmentations_path = segmentations_path

        self.labels_dict = create_labels_df(self.labels_path, echogenicity)

        self.img_files = sorted(listdir(images_path))
        self.segmentation_files = sorted(listdir(segmentations_path))
        self.complete_images = []
        # print(self.segmentation_files)
        # print(self.segmentation_files)
        for segmentation_file in self.segmentation_files:
            if segmentation_file in self.img_files and segmentation_file in self.labels_dict:
                self.complete_images.append(segmentation_file)
        # print(len(self.complete_images))
        self.transformations_custom = transformations_custom
        self.transformations_torch = transformations_torch
        # self.transformations_torch_label = transformations_torch_label

    def __getitem__(self, index: int) -> tuple:
        """Gets item from the dataset at a specified index.

        Parameters
        ----------
        index : int
            Index of an item to return.

        Returns
        -------
        tuple
            Returns the processed image and the transformed segmentation mask
        """
        img = load_img(self.images_path, self.complete_images[index])
        segmentation = load_img(self.segmentations_path, self.complete_images[index])
        label = self.labels_dict[self.complete_images[index]]
        img, segmentation = self.transformations_custom(img, segmentation)
        img_tensor = self.transformations_torch(img)
        
        segmentation_tensor = self.transformations_torch(segmentation)
        segmentation_mask = label_to_mask(segmentation_tensor, encode_to_one_hot=False)
        

        # return {
        #     "img_name":self.complete_images[index],
        #     "img": img_tensor,
        #     "segmentation": segmentation_mask,
        #     "echogenicity": label[0],
        #     "homogenicity": label[1]
        # }
        # plot_image(img_tensor)
        a = self.masking_function(img_tensor.clone(), segmentation_mask)
        # if label == 1:
        #     k = 0.9
        # elif label == 0:
        #     k = 0.1
        # result_tensor = torch.where(a != 0, k * torch.ones_like(a), a)
        # nonzero_indices = np.nonzero(a)
        # print(nonzero_indices)
        # # Get the non-zero values from the original tensor
        # # nonzero_values = a[nonzero_indices]
        # print(a[0, nonzero_indices[0][1], nonzero_indices[0][2]])
        # print(a[1, nonzero_indices[0][1], nonzero_indices[0][2]])
        # print(a[2, nonzero_indices[0][1], nonzero_indices[0][2]])
        # print(a.shape)
        # plot_image(a)
        # plot_image_label(img_tensor, segmentation_mask)

        ### DELETE AFTER TESTING !!!!!
        # return result_tensor, tensor(label).long(), self.complete_images[index]
    
        return a, tensor(label).long(), self.complete_images[index]

    def __len__(self) -> int:
        """Returns a length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.complete_images)
    
    

class NonGeomParameterDataset_for_visualizing(Dataset):

    def __init__(
        self,
        images_path: str,
        segmentations_path: str,
        labels_path: str,
        masking_function,
        echogenicity,
        transformations_torch: Compose,
        transformations_custom: SegCompose,
        
    ) -> None:
        self.masking_function = masking_function
        self.images_path = images_path
        self.labels_path = labels_path
        self.segmentations_path = segmentations_path

        self.labels_dict = create_labels_df(self.labels_path, echogenicity)

        self.img_files = sorted(listdir(images_path))
        self.segmentation_files = sorted(listdir(segmentations_path))
        self.complete_images = []
        # print(self.segmentation_files)
        # print(self.segmentation_files)
        for segmentation_file in self.segmentation_files:
            if segmentation_file in self.img_files and segmentation_file in self.labels_dict:
                self.complete_images.append(segmentation_file)
        # print(len(self.complete_images))
        self.transformations_custom = transformations_custom
        self.transformations_torch = transformations_torch
        # self.transformations_torch_label = transformations_torch_label

    def __getitem__(self, index: int) -> tuple:
        """Gets item from the dataset at a specified index.

        Parameters
        ----------
        index : int
            Index of an item to return.

        Returns
        -------
        tuple
            Returns the processed image and the transformed segmentation mask
        """
        img = load_img(self.images_path, self.complete_images[index])
        segmentation = load_img(self.segmentations_path, self.complete_images[index])
        label = self.labels_dict[self.complete_images[index]]
        img, segmentation = self.transformations_custom(img, segmentation)
        img_tensor = self.transformations_torch(img)
        
        segmentation_tensor = self.transformations_torch(segmentation)
        segmentation_mask = label_to_mask(segmentation_tensor, encode_to_one_hot=False)
        

        # return {
        #     "img_name":self.complete_images[index],
        #     "img": img_tensor,
        #     "segmentation": segmentation_mask,
        #     "echogenicity": label[0],
        #     "homogenicity": label[1]
        # }
        # plot_image(img_tensor)
        a = self.masking_function(img_tensor.clone(), segmentation_mask)
        # if label == 1:
        #     k = 0.9
        # elif label == 0:
        #     k = 0.1
        # result_tensor = torch.where(a != 0, k * torch.ones_like(a), a)
        # nonzero_indices = np.nonzero(a)
        # print(nonzero_indices)
        # # Get the non-zero values from the original tensor
        # # nonzero_values = a[nonzero_indices]
        # print(a[0, nonzero_indices[0][1], nonzero_indices[0][2]])
        # print(a[1, nonzero_indices[0][1], nonzero_indices[0][2]])
        # print(a[2, nonzero_indices[0][1], nonzero_indices[0][2]])
        # print(a.shape)
        # plot_image(a)
        # plot_image_label(img_tensor, segmentation_mask)

        ### DELETE AFTER TESTING !!!!!
        # return result_tensor, tensor(label).long(), self.complete_images[index]
    
        return a, tensor(label).long(), self.complete_images[index], img_tensor, segmentation_tensor, segmentation_mask 

    def __len__(self) -> int:
        """Returns a length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.complete_images)
    

class SyntEchoDataset(Dataset):

    def __init__(
        self,
        images_path: str,
        segmentations_path: str,
        masking_function,
        transformations_torch: Compose,
        transformations_custom: SegCompose,
        
    ) -> None:
        self.masking_function = masking_function
        self.images_path = images_path
        self.segmentations_path = segmentations_path


        self.img_files = sorted(listdir(images_path))
        self.segmentation_files = sorted(listdir(segmentations_path))
        self.complete_images = []
        for segmentation_file in self.segmentation_files:
            if segmentation_file in self.img_files and segmentation_file:
                self.complete_images.append(segmentation_file)
        self.transformations_custom = transformations_custom
        self.transformations_torch = transformations_torch

    def __getitem__(self, index: int) -> tuple:
        """Gets item from the dataset at a specified index.

        Parameters
        ----------
        index : int
            Index of an item to return.

        Returns
        -------
        tuple
            Returns the processed image and the transformed segmentation mask
        """
        img = load_img(self.images_path, self.complete_images[index])
        segmentation = load_img(self.segmentations_path, self.complete_images[index])
        label = int(self.complete_images[index][0])
        img, segmentation = self.transformations_custom(img, segmentation)
        img_tensor = self.transformations_torch(img)
        
        segmentation_tensor = self.transformations_torch(segmentation)
        segmentation_mask = label_to_mask(segmentation_tensor, encode_to_one_hot=False)
        
        # plot_image(img_tensor)
        # print(self.complete_images[index])
        # print(label)
        # plot_image(img_tensor)
        a = self.masking_function(img_tensor.clone(), segmentation_mask)
        # if label == 1:
        #     k = 0.9
        # elif label == 0:
        #     k = 0.1
        # result_tensor = torch.where(a != 0, k * torch.ones_like(a), a)
        # nonzero_indices = np.nonzero(a)
        # print(nonzero_indices)
        # # Get the non-zero values from the original tensor
        # # nonzero_values = a[nonzero_indices]
        # print(a[0, nonzero_indices[0][1], nonzero_indices[0][2]])
        # print(a[1, nonzero_indices[0][1], nonzero_indices[0][2]])
        # print(a[2, nonzero_indices[0][1], nonzero_indices[0][2]])
        # print(a.shape)
        # plot_image(a)
        # plot_image_label(img_tensor, segmentation_mask)


        ### DELETE AFTER TESTING !!!!!
        # return result_tensor, tensor(label).long(), self.complete_images[index]
    
        return a, tensor(label).long(), self.complete_images[index]

    def __len__(self) -> int:
        """Returns a length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.complete_images)