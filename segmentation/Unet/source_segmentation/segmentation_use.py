from os import listdir, path
from carotids.segmentation.visualization import plot_segmentation_prediction, plot_segmentation_prediction_differences
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from torch import device, load, save
import torch.nn as nn
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
)

from carotids.preprocessing import load_img
from carotids.segmentation.model_archive import Unet
CATEGORIES = 3
DEVICE = device("cpu")
TRANSFORMATIONS_TORCH = Compose(
    [
        Resize((512, 512)),
        ToTensor(),
    ]
)

PATH_TO_TRANS_MODEL = path.join("models", "transverse_segmentation_model.pt")
PATH_TO_TRANS_DATA = path.join("data_samples", "segmentation_samples", "transverse")

PATH_TO_LONG_MODEL = path.join("models", "longitudinal_segmentation_model.pt")
PATH_TO_LONG_DATA = path.join("data_samples", "segmentation_samples", "longitudinal")


def segmentation_example_use(path_to_model, path_to_data):
    """Plots segmentation masks of images specified by path_to_data
    created by the model saved in path_to_model.

    Parameters
    ----------
    path_to_model : str
        Path to the model.
    path_to_data : str
        Path to the data.
    """
    model = Unet(CATEGORIES)
    # model = UnetDVCFS(CATEGORIES)
    # model = nn.DataParallel(model)
    # state_dict = torch.load(path_to_model,map_location=device)
    # # create new OrderedDict that does not contain `module.`
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:] #remove 'module'
    #     new_state_dict[name] = v


    model.load_state_dict(load(path_to_model, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    image_names = sorted(listdir(path_to_data))

    for image_name in image_names:
        print(f"Image: {image_name}")
        img = load_img(path_to_data, image_name)
        img_tensor = TRANSFORMATIONS_TORCH(img)
        img_tensor.to(DEVICE)

        mask = model(img_tensor.unsqueeze(0))
        prediction = mask.squeeze().detach().cpu().numpy().argmax(0) / CATEGORIES
        print(mask.shape)
        print(mask)
        plt.imshow(
            prediction, cmap=cm.gray
        )
        plt.show()


        plot_segmentation_prediction_differences()


        break

# from torchsummary import summary
if __name__ == "__main__":
    print("Transverse data...")
    segmentation_example_use(PATH_TO_TRANS_MODEL, PATH_TO_TRANS_DATA)

    # Load the model from the .pt file
    # model = torch.load(PATH_TO_TRANS_MODEL, map_location=torch.device('cpu'))
    # for k, v in model.items():
    #     print(k)
    # print('-----------------------------------')
    # model2 = Unet(CATEGORIES)
    # for name, param in model2.named_parameters():
    #     print(name)  # This will print the name of the parameter
    #     # print(param)  # This will print the parameter tensor itself

    # Print the model architecture
    # print(model.)
    #print("Longitudinal data...")
    #segmentation_example_use(PATH_TO_LONG_MODEL, PATH_TO_LONG_DATA)
