from os import listdir, path
from PIL import Image
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from torch import device
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
)

from segmentation.common.preprocessing import load_img

TRANSFORMATIONS_TORCH = Compose(
    [
        Resize((256, 256)),
        ToTensor(),
    ]
)

def create_seg_image(segmentation_output):
    color_mapping = {
        0: [0, 0, 0],   # Background (black)
        1: [255, 0, 0], # wall (red)
        2: [0, 255, 0], # plaque (green)
        3: [0, 0, 255], # lumen (blue)
    }
    segmentation_output = segmentation_output.squeeze()
    segmentation_np = segmentation_output.detach().cpu().numpy()
    predicted_class = np.argmax(segmentation_np, axis=0)
    output_image = np.zeros((segmentation_output.shape[1], segmentation_output.shape[2], 3), dtype=np.uint8)

    for class_idx, color in color_mapping.items():
        output_image[predicted_class == class_idx] = color
    return Image.fromarray(output_image)

def create_segmentations(model, path_input_folder, path_output_folder, device = device("cpu"), transformation_torch = TRANSFORMATIONS_TORCH):
    """
    Performs image segmentation on all images in a given input folder using a specified model, 
    and saves the segmented images in an output folder. The function allows for the model to 
    run on a specified device (e.g., CPU or GPU) and applies a given transformation to each 
    image before segmentation.

    Args:
        model (torch.nn.Module): The pre-trained segmentation model used for processing the images. 
                                This model should take an image tensor as input and return a 
                                segmentation map.
        path_input_folder (str): Path to the input folder containing the images to be segmented. 
                                The function processes all images in this folder.
        path_output_folder (str): Path to the output folder where the segmented images will be saved. 
                                Each output image is saved with the same filename as the input image.
        device (torch.device, optional): The device (CPU or GPU) on which the model will execute. 
                                        Defaults to torch.device("cpu").
        transformation_torch (callable, optional): A transformation function to be applied to each 
                                                image before feeding it to the model. This function 
                                                should be compatible with PyTorch's transformation 
                                                methods. Defaults to TRANSFORMATIONS_TORCH, which 
                                                should be predefined.
"""
    model.to(device)
    model.eval()
    image_names = sorted(listdir(path_input_folder))
    for image_name in image_names:
        print(f"segmenting: {image_name}")
        img = load_img(path_input_folder, image_name)
        img_tensor = transformation_torch(img)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(device)

        prediction = model(img_tensor)
        output_image_pil = create_seg_image(prediction)
        output_image_pil.save(path.join(path_output_folder, image_name))

