from json import load
from os import listdir

from numpy import asarray
from torch import as_tensor, int64, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose

from localization.preprocessing import load_img



class FastCarotidDatasetEval(Dataset):
    """Represents a dateset used to store data which are the input into
    Faster R-CNN. Reads all files in a folder. The data are loaded when an
    item is gotten.
    """

    def __init__(self, data_path: int, transformations: Compose) -> None:
        """Initializes a training Faster R-CNN dataset.

        Parameters
        ----------
        data_path : str
            Path to a folder containing the images.
        transformations : Compose
            List of transformations used to preprocess the image inputs.
        """
        self.data_path = data_path
        self.data_files = sorted(listdir(data_path))
        self.transformations = transformations

    def __getitem__(self, index: int) -> tuple:
        """Gets item from the dataset at a specified index.

        Parameters
        ----------
        index : int
            Index of an item to return.

        Returns
        -------
        tuple
            Image processed into a tensor and name of the image file.
        """
        img = load_img(self.data_path, self.data_files[index])

        return self.transformations(img), self.data_files[index]

    def __len__(self) -> int:
        """Returns a length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.data_files)
