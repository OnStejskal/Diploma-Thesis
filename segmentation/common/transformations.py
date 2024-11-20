from numpy import asarray, where
from numpy.random import normal, randint, uniform

from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_gamma,
    crop,
    hflip,
    rotate,
    vflip,
)


class SegRandomRotation:
    """Random rotation.

    Represents random rotation which transforms image and segmentation mask.
    """

    def __init__(self, p: float = 0.5, min_angle: int = -10, max_angle: int = 10):
        """Initializes a random rotation.

        Parameters
        ----------
        p : float
            Probability with which transformation is applied.
        min_angle : int
            Minimal angle bound for rotation (inclusive).
        max_angle : int
            Max angle bound for rotation (exclusive).
        """
        self.p = p
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, img: Image, mask: Tensor) -> tuple:
        """Applies random rotation on an image and a mask.

        Parameters
        ----------
        img : Image
            Image to transform.
        mask : Tensor
            Mask to transform.

        Returns
        -------
        tuple
            Transformed image and mask.
        """
        if uniform() <= self.p:
            rotation_angle = randint(self.min_angle, self.max_angle)
            return rotate(img, rotation_angle), rotate(mask, rotation_angle)

        return img, mask


class SegRandomContrast:
    """Random contrast.

    Represents random contrast which transforms image.
    """

    def __init__(self, p: float = 0.5, mean: float = 1.0, std: float = 0.025):
        """Initializes a random contrast.

        Parameters
        ----------
        p : float
            Probability with which transformation is applied.
        mean : float
            Mean of the contrast factor distribution for transformation.
        std : float
            Standard deviance of the contrast factor distribution for
            transformation.
        """
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, img: Image, mask: Tensor) -> tuple:
        """Applies contrast on an image.

        Parameters
        ----------
        img : Image
            Image to transform.
        mask : Tensor
            Mask to pass.

        Returns
        -------
        tuple
            Transformed image and original mask.
        """
        if uniform() <= self.p:
            contrast_factor = max(0, normal(self.mean, self.std))
            return adjust_contrast(img, contrast_factor), mask

        return img, mask


class SegRandomGammaCorrection:
    """Random gamma correction.

    Represents random gamma correction which transforms image.
    """

    def __init__(self, p: float = 0.5, mean: float = 1.0, std: float = 0.025):
        """Initializes a random gamma correction.

        Parameters
        ----------
        p : float
            Probability with which transformation is applied.
        mean : float
            Mean of the gamma correction distribution for transformation.
        std : float
            Standard deviance of the gamma correction distribution for
            transformation.
        """
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, img: Image, mask: Tensor) -> tuple:
        """Applies gamma correction on an image.

        Parameters
        ----------
        img : Image
            Image to transform.
        mask : Tensor
            Mask to pass.

        Returns
        -------
        tuple
            Transformed image and original mask.
        """
        if uniform() <= self.p:
            gamma = max(0, normal(self.mean, self.std))
            return adjust_gamma(img, gamma), mask

        return img, mask


class SegRandomBrightness:
    """Random brightness.

    Represents random brightness which transforms image.
    """

    def __init__(self, p: float = 0.5, mean: float = 1.0, std: float = 0.025):
        """Initializes a random brightness.

        Parameters
        ----------
        p : float
            Probability with which transformation is applied.
        mean : float
            Mean of the brightness factor distribution for transformation.
        std : float
            Standard deviance of the brightness factor distribution for
            transformation.
        """
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, img: Image, mask: Tensor) -> tuple:
        """Applies brightness on an image.

        Parameters
        ----------
        img : Image
            Image to transform.
        mask : Tensor
            Mask to pass.

        Returns
        -------
        tuple
            Transformed image and original mask.
        """
        if uniform() <= self.p:
            brightness_factor = max(0, normal(self.mean, self.std))
            return adjust_brightness(img, brightness_factor), mask

        return img, mask


class SegRandomHorizontalFlip:
    """Random horizontal flip.

    Represents random horizontal flip which transforms image and segmentation mask.
    """

    def __init__(self, p: float = 0.5):
        """Initializes a random horizontal flip.

        Parameters
        ----------
        p : float
            Probability with which transformation is applied.
        """
        self.p = p

    def __call__(self, img: Image, mask: Tensor) -> tuple:
        """Applies random horizontal flip on an image and a mask.

        Parameters
        ----------
        img : Image
            Image to transform.
        mask : Tensor
            Mask to transform.

        Returns
        -------
        tuple
            Transformed image and mask.
        """
        if uniform() <= self.p:
            return hflip(img), hflip(mask)

        return img, mask


class SegRandomVerticalFlip:
    """Random vertical flip.

    Represents random vertical flip which transforms image and segmentation mask.
    """

    def __init__(self, p: float = 0.5):
        """Initializes a random vertical flip.

        Parameters
        ----------
        p : float
            Probability with which transformation is applied.
        """
        self.p = p

    def __call__(self, img: Image, mask: Image) -> tuple:
        """Applies random vertical flip on an image and a mask.

        Parameters
        ----------
        img : Image
            Image to transform.
        mask : Image
            Mask to transform.

        Returns
        -------
        tuple
            Transformed image and mask.
        """
        if uniform() <= self.p:
            return vflip(img), vflip(mask)

        return img, mask


class SegCrop:
    """Random crop.

    Represents random crop which transforms image and segmentation mask.
    Crops an input image so that whole object is perserved. This is done
    by adding a random number of pixels to each side. Also has a default
    setting that adds fixed number of pixels to each side.
    """

    def __init__(self, random_t: int = 0, default_t: int = 5, default_size = None, square_output = False):
        """Initializes a random crop.

        Parameters
        ----------
        random_t : int
            Upper bound for number of pixels added.
        default_t : int
            Fixed number of pixels added to a side.
        """
        self.random_t = random_t
        self.default_t = default_t
        self.default_size = default_size
        self.square_output = square_output

    def __call__(self, img: Image, mask: Image) -> tuple:
        """Applies random crop on an image and a mask.

        Parameters
        ----------
        img : Image
            Image to transform.
        mask: Image
            Mask to transform.

        Returns
        -------
        tuple
            Transformed image and mask.
        """
        
        mask_indices = where((asarray(mask) == [255, 255, 255]).any(axis=2))
        x1, y1 = mask_indices[1].min(), mask_indices[0].min()
        x2, y2 = mask_indices[1].max(), mask_indices[0].max()
        # print(img.size)
        # print(x1)
        # print(y1)
        # print(x2)
        # print(y2)
        width, height = img.size

        if self.random_t != 0:
            t = randint(self.random_t)

            x1, y1 = max(x1 - t, 0), max(y1 - t, 0)
            x2, y2 = min(x2 + t, width), min(y2 + t, height)
        else:
            t = self.default_t

            x1, y1 = max(x1 - t, 0), max(y1 - t, 0)
            x2, y2 = min(x2 + t, width), min(y2 + t, height)

        # print(x1)
        # print(y1)
        # print(x2)
        # print(y2)
        if self.square_output:
            width = x2 - x1
            height = y2 - y1
            if width >= height:
                diff = width - height
                y1 = y1 - diff//2
                y2 = y2 + diff // 2 + (diff%2)
            else:
                diff = height - width
                x1 = x1 - diff//2
                x2 = x2 + diff // 2 + (diff%2)


        if self.default_size is not None:
            crop_center_x = (x1 + x2) // 2
            crop_center_y = (y1 + y2) // 2

            # Size of the new crop
            new_width, new_height = self.default_size

            # Calculate new bounding box coordinates with the crop center being the center of the new crop
            new_x1 = max(crop_center_x - new_width // 2, 0)
            new_x2 = min(crop_center_x + new_width // 2, img.width)
            new_y1 = max(crop_center_y - new_height // 2, 0)
            new_y2 = min(crop_center_y + new_height // 2, img.height)

            final_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))
            final_seg = Image.new('RGB', (new_width, new_height), (0, 0, 0))
            cropped_image = img.crop((new_x1, new_y1, new_x2, new_y2))
            paste_x = (new_width - (new_x2 - new_x1)) // 2
            paste_y = (new_height - (new_y2 - new_y1)) // 2
            final_image.paste(cropped_image, (paste_x, paste_y))

            final_seg = Image.new('RGB', (new_width, new_height), (0, 0, 0))
            cropped_seg = mask.crop((new_x1, new_y1, new_x2, new_y2))
            final_seg.paste(cropped_seg, (paste_x, paste_y))
            return final_image, final_seg


        return img.crop((x1, y1, x2, y2)), mask.crop((x1, y1, x2, y2))


class SegCompose:
    def __init__(self, transformations: list = []):
        """Initializes a composition of custom segmentation transformations.

        Parameters
        ----------
        transformations: list
            List of transformations to apply.
        """
        self.transformations = transformations

    def __call__(self, img: Image, mask: Image) -> tuple:
        """Applies transformations on an image and a mask.

        Parameters
        ----------
        img : Image
            Image to be processed.
        mask : Image
            Mask to be processed.

        Returns
        -------
        tuple
            Processed image and mask.
        """
        for transformation in self.transformations:
            img, mask = transformation(img, mask)

        return img, mask
