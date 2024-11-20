from torch import device, load, no_grad
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from os.path import join
from localization.frcnn_dataset import FastCarotidDatasetEval
from localization.models import create_faster_rcnn
from PIL import Image
from utils import create_squared_crop_coordinates_img, create_fix_crop_coordinates_img
from segmentation.common.visualization import plot_image
import matplotlib.pyplot as plt
TRANSFORMATIONS = Compose([ToTensor()])

PATH_TO_TRANS_MODEL = join("model", "transverse_localization_model.pt")
PATH_TO_TRANS_DATA = join("data")


@no_grad()
def create_localizations(path_to_model: str, path_to_data: str, path_to_output: str, device = device("cpu"), min_score_to_pass = 0, square = False, fix = False) -> None:
    """Example usage of localization model. Load model from path selected by
    parameter path_to_model. Evaluates the images in the folder specified by
    the path_to_data parameter. Prints name of the file and coordinates of the
    most probable carotid on the image with its score.

    Parameters
    ----------
    path_to_model : str
        Path to the model.
    path_to_data : str
        Path to the data.
    """
    model = create_faster_rcnn()
    model.load_state_dict(load(path_to_model, map_location=device))
    model.to(device)
    model.eval()

    dataset = FastCarotidDatasetEval(path_to_data, TRANSFORMATIONS)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for image_tensor, image_name in loader:
        image_name = image_name[0]
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        print(outputs)
        if not outputs[0]["boxes"].numel() == 0:
            if outputs[0]["scores"][0] >= min_score_to_pass:
                print(image_name)
                original_image = Image.open(join(path_to_data, image_name))
                coordinates = outputs[0]["boxes"][0].detach().numpy()
                # print(coordinates)
                if fix:
                    cropped_image = create_fix_crop_coordinates_img(coordinates, original_image)
                elif square:
                    cropped_image = create_squared_crop_coordinates_img(coordinates, original_image)
                else:            
                    cropped_image = original_image.crop(coordinates)
                # print(cropped_image.size)
                # plt.imshow(cropped_image)
                # plt.axis('off')  # Turn off axis labels and ticks
                # plt.show()
                cropped_image.save(join(path_to_output, image_name))
        else:
            print("no output")