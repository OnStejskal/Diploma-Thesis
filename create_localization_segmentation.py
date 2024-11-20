import argparse
import sys
from segmentation.use_segmentor import create_segmentations
from localization.localization_use import create_localizations
from os.path import join
import pandas as pd
import os
from utils import create_folder_structure_for_parameters_computation
import numpy as np
from torch import device, load, save, cuda
from PIL import Image
# from segmentation.Unet.model import Unet, UnetDVCFS
from segmentation.ACNN.segmentor import Unet_Dropout, UnetDVCFS, Unet
from segmentation.ACNN.segmentor_archive import Unet as Unet2
from segmentation.MAAG.architectures.segmentor_UNET import Segmentor_with_AG
from segmentation.SEGan.net import NetS
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
)
from os import makedirs
CATEGORIES = 4
DO_LOCALIZATION = True
DO_SEGMENTATION = True

INPUT_DATASET_NAME = "carotid_key_inpainted_dataset" # folder name which includes folder "images" with the raw ultrasound images of the carotid


################# For localization and segmentation data please fill specification #################
EXPERIMENT_NAME = "Fill the name of the experiment"
TYPE_NETWORK = "fill the type of the model" # Either maag, acnn_seq, acnn_sim, segan, unet, maag_weak
SEGMENTATION_EXPERIMENT_NAME = "Fill the name of the directory where is the model"
MODEL = "initialize the segmentor mdoel for given method" # can be found below in the examples
IS_FIX = "fill bool" # True for Fixed-size models, False for variable-size models 


################# Exmaples for all the types used in thesis #################

# IS_FIX = False
# EXPERIMENT_NAME = "MAAG_DYNAMIC"
# TYPE_NETWORK = "maag"
# SEGMENTATION_EXPERIMENT_NAME = "maag_without_unsup_gen_loss"
# MODEL = Segmentor_with_AG(number_of_classes_with_background=CATEGORIES)


# IS_FIX = False
# EXPERIMENT_NAME = "ACNN_DYNAMIC"
# TYPE_NETWORK = "acnn_sim"
# SEGMENTATION_EXPERIMENT_NAME = "aelabelswapboth_acnn_minst_dp_nojointae_dynamic_euclid_weight"
# MODEL = Unet2(CATEGORIES)


# IS_FIX = False
# EXPERIMENT_NAME = "SEGAN_DYNAMIC"
# TYPE_NETWORK = "segan"
# SEGMENTATION_EXPERIMENT_NAME = "new_sg_rectangle_b8_lr2e-5_clamp1e-2_alpha1e-1"
# MODEL = NetS(classes=CATEGORIES)

# IS_FIX = True
# EXPERIMENT_NAME = "MAAG_FIX"
# TYPE_NETWORK = "maag_weak"
# SEGMENTATION_EXPERIMENT_NAME = "maag_weak_fix_unsup05_moreep"
# MODEL = Segmentor_with_AG(number_of_classes_with_background=CATEGORIES)

# IS_FIX = True
# EXPERIMENT_NAME = "ACNN_FIX"
# TYPE_NETWORK = "acnn_seq"
# SEGMENTATION_EXPERIMENT_NAME = "minst_sequential_rectangle_256_fix"
# MODEL = Unet_Dropout(CATEGORIES)

# IS_FIX = True
# EXPERIMENT_NAME = "SEGAN_FIX"
# TYPE_NETWORK = "segan"
# SEGMENTATION_EXPERIMENT_NAME = "new_sg_rectangle_b8_lr2e-5_clamp1e-2_alpha1e-1"
# MODEL = NetS(classes=CATEGORIES)

# IS_FIX = True
# EXPERIMENT_NAME = "UNETPAPER_FIX"
# TYPE_NETWORK = "unet"
# SEGMENTATION_EXPERIMENT_NAME = "fix_unet_paper"
# MODEL = UnetDVCFS(CATEGORIES)
################# ##################### #################



SEGMENTATION_MODEL_PATH = join("segmentation", "models", TYPE_NETWORK,SEGMENTATION_EXPERIMENT_NAME,f"{SEGMENTATION_EXPERIMENT_NAME}.pt")

TRANSFORMATIONS_TORCH = Compose(
    [
        Resize((256, 256)),
        ToTensor(),
    ]
)

CREATE_FOLDER_FOR_PARAMETER_COMPUTATION = True

def create_localization_segmentation(torch_device, do_localization = False, do_segmentation = True, create_folder_for_parameter_computation = True):
    input_image_folder = join("data", INPUT_DATASET_NAME, "images")
    localized_folder = join("data", INPUT_DATASET_NAME, EXPERIMENT_NAME, "localized")
    segmentation_folder = join("data", INPUT_DATASET_NAME, EXPERIMENT_NAME, "segmentations")
    
    ################### LOCALIZE IMAGES ##################
    if do_localization:
        makedirs(localized_folder, exist_ok=True)
        create_localizations(join("localization", "model", "transverse_localization_model.pt"), input_image_folder, localized_folder,min_score_to_pass=0.9, square=False, fix = IS_FIX)
        input_image_folder = localized_folder
    else:
        pass #localized_folder = input_image_folder

    
    ################### SEGMENT IMAGES ##################
    if do_segmentation:
        makedirs(segmentation_folder, exist_ok=True)
        MODEL.load_state_dict(load(SEGMENTATION_MODEL_PATH, map_location=torch_device))
        create_segmentations(MODEL, localized_folder, segmentation_folder, device=torch_device,transformation_torch=TRANSFORMATIONS_TORCH)
    
    if create_folder_for_parameter_computation:
        output_dir_path = join("parameters_computation", "data", EXPERIMENT_NAME)
        if os.path.exists(output_dir_path):
            print("DIRECTORY ALREADY EXIST")
        create_folder_structure_for_parameters_computation(localized_folder, segmentation_folder, output_dir_path)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, help="number of CUDA device")
    args = parser.parse_args()
    if args.device or args.device == 0:
        DEVICE_NUMBER = args.device
        print(f"DEVICE NUMBER is set to {args.device}")
    else:
        print("No number was provided. Please Provide number")
        sys.exit()

    if cuda.is_available():
        print("CUDA IS AVAILABLE, DEVICE NUMBER {}".format(DEVICE_NUMBER))
        DEVICE = device(DEVICE_NUMBER)
        cuda.set_device(DEVICE_NUMBER)
    else:
        print("NO CUDA IS AVAILABLE, TRAINING ON CPU")
        DEVICE = device("cpu")

    if os.path.exists(join("data", INPUT_DATASET_NAME, EXPERIMENT_NAME)):
        user_input = input(f"experiment: {EXPERIMENT_NAME} exist do you want to continue (y/n): ").lower()
        if not(user_input == 'y' or user_input == 'yes'):
            print("Exiting the program.")
            sys.exit()
    else:
        print(f"creating {EXPERIMENT_NAME} experiment")
    makedirs(join("data", INPUT_DATASET_NAME, EXPERIMENT_NAME), exist_ok=True)

    if CREATE_FOLDER_FOR_PARAMETER_COMPUTATION and  os.path.exists(join("parameters_computation", "data", EXPERIMENT_NAME)):
        user_input = input(f"train/val parameter_computation data for: {EXPERIMENT_NAME} exist do you want to continue (y/n): ").lower()
        if not(user_input == 'y' or user_input == 'yes'):
            print("Exiting the program.")
            sys.exit()
    else:
        print(f"creating {EXPERIMENT_NAME} experiment")
    makedirs(join("data", INPUT_DATASET_NAME, EXPERIMENT_NAME), exist_ok=True)


    create_localization_segmentation(DEVICE, do_localization=DO_LOCALIZATION, do_segmentation= DO_SEGMENTATION, create_folder_for_parameter_computation = CREATE_FOLDER_FOR_PARAMETER_COMPUTATION)