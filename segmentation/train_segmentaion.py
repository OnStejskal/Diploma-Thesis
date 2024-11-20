import argparse
import sys
from os.path import join, exists
from os import makedirs
from torch import cuda, device
import torch
from MAAG.train_model_supervised import maag_run_segmentation
from MAAG.train_model_with_weak import maag_run_segmentation_with_weak
from Unet.train_model import unet_run_segmentation
from SEGan.train_model_s3c3 import segan_run_segmentation
from ACNN.train_simulaneously import accn_sim_run_training_segmentation
from ACNN.train_sequentially import accn_seq_run_training_segmentation
from ACNN.train_no_ae import accn_noae_run_training_segmentation
from common.transformations import (
    SegCompose,
    SegCrop,
    SegRandomRotation,
    SegRandomVerticalFlip,
    SegRandomHorizontalFlip,
    SegRandomBrightness,
    SegRandomContrast,
    SegRandomGammaCorrection
)
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    CenterCrop
)
from common.visualization import create_train_val_graphs
from json import dump
from test_segmentor import test_segmentor


## Types of networks: ["unet", "segan", "maag", "acnn_seq", "acnn_sim"] ##    
################ FILL THESE ################
TYPE_OF_NETWORK = "fill the method for segmentation" ## Types of networks: ["unet", "segan", "maag", "acnn_seq", "acnn_sim"] ##    
EXPERIMENT_NAME = "fill the experiment name"

TRAIN_IMG_PATH = "fill the path"
TRAIN_LABELS_PATH = "fill the path"
VAL_IMG_PATH = "fill the path"
VAL_LABELS_PATH = "fill the path"
TEST_IMG_PATH = "fill the path"
TEST_LABELS_PATH = "fill the path"

EPOCHS = "fill the number of epochs"
CATEGORIES = "fill the number of categories"
PLAQUE_WITH_WALL = "fill True or False"
IMAGE_SIZE_IN_PIXELS = "fill the shape tupl"
MODEL_INPUT_SHAPE = "fill the shape tupl"

TRAIN_TRANSFORMATIONS_SEG = "fill the transformations for training"
VAL_TRANSFORMATIONS_SEG = "fill the transformations for validation"

################ EXAMPLE ################
TYPE_OF_NETWORK = "maag" ## Types of networks: ["unet", "segan", "maag", "acnn_seq", "acnn_sim"] ##    
EXPERIMENT_NAME = "maag_fixed_dataset"

TRAIN_IMG_PATH = join("data", "train_val_test", "train", "images")
TRAIN_LABELS_PATH = join("data", "train_val_test", "train", "segmentations")
VAL_IMG_PATH = join("data", "train_val_test", "val", "images")
VAL_LABELS_PATH = join("data", "train_val_test","val", "segmentations")
TEST_IMG_PATH = join("data", "train_val_test","test", "images")
TEST_LABELS_PATH = join("data", "train_val_test","test", "segmentations")

EPOCHS = 300
CATEGORIES = 4
PLAQUE_WITH_WALL = False
IMAGE_SIZE_IN_PIXELS = (380, 380)
MODEL_INPUT_SHAPE = (256, 256)

TRAIN_TRANSFORMATIONS_SEG = SegCompose(
    [
        SegRandomRotation(),
        SegRandomContrast(),
        SegRandomBrightness(),
        SegRandomHorizontalFlip(),
        SegCrop(default_t=15, default_size=IMAGE_SIZE_IN_PIXELS), #if you want variable size dont fill the defaults size
    ]
)
VAL_TRANSFORMATIONS_SEG = SegCompose(
    [
        SegCrop(default_t=15, default_size=IMAGE_SIZE_IN_PIXELS) #if you want variable size dont fill the defaults size
    ]
)

# additional dataset for the use of weak annotatiaons
WEAK_IMG_PATH = join("data", "weak_annotations", "images")
WEAK_LABELS_PATH = join("data", "weak_annotations", "segmentations")
################ ################ ################ ################ ################ ################ ################








EXPERIMENT_PATH = join("models", TYPE_OF_NETWORK, EXPERIMENT_NAME)

TRANSFORMATIONS_TORCH = Compose(
    [
        Resize(MODEL_INPUT_SHAPE),
        ToTensor(),
    ]
)

WEAK_ANNOTATIONS_TRANSFORMATIONS_TORCH = Compose(
    [
        CenterCrop(IMAGE_SIZE_IN_PIXELS),
        Resize(MODEL_INPUT_SHAPE),
        ToTensor(),
    ]
)

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
    
    if exists(EXPERIMENT_PATH):
        user_input = input(f"experiment: {EXPERIMENT_PATH} exist do you want to continue (y/n): ").lower()
        if not(user_input == 'y' or user_input == 'yes'):
            print("Exiting the program.")
            sys.exit()

    makedirs(EXPERIMENT_PATH,exist_ok = True)
    print("experimetn name: ", EXPERIMENT_NAME)
    print("architecture: ", TYPE_OF_NETWORK)
    print("epochs: ", EPOCHS)
    if TYPE_OF_NETWORK == "unet":
        model, train_results = unet_run_segmentation(
            DEVICE,
            TRAIN_IMG_PATH,
            TRAIN_LABELS_PATH,
            VAL_IMG_PATH,
            VAL_LABELS_PATH,
            TRAIN_TRANSFORMATIONS_SEG,
            VAL_TRANSFORMATIONS_SEG,
            TRANSFORMATIONS_TORCH,
            PLAQUE_WITH_WALL,
            CATEGORIES,
            EPOCHS
        )
    elif TYPE_OF_NETWORK == "segan":
        model, train_results = segan_run_segmentation(
            DEVICE,
            TRAIN_IMG_PATH,
            TRAIN_LABELS_PATH,
            VAL_IMG_PATH,
            VAL_LABELS_PATH,
            TRAIN_TRANSFORMATIONS_SEG,
            VAL_TRANSFORMATIONS_SEG,
            TRANSFORMATIONS_TORCH,
            PLAQUE_WITH_WALL,
            CATEGORIES,
            EPOCHS
        )
    elif TYPE_OF_NETWORK == "maag":    
        model, train_results = maag_run_segmentation(
            DEVICE,
            TRAIN_IMG_PATH=TRAIN_IMG_PATH,
            TRAIN_LABELS_PATH=TRAIN_LABELS_PATH,
            VAL_IMG_PATH = VAL_IMG_PATH,
            VAL_LABELS_PATH = VAL_LABELS_PATH,
            TRAIN_TRANSFORMATIONS_SEG = TRAIN_TRANSFORMATIONS_SEG,
            VAL_TRANSFORMATIONS_SEG = VAL_TRANSFORMATIONS_SEG,
            TRANSFORMATIONS_TORCH =TRANSFORMATIONS_TORCH,
            PLAQUE_WITH_WALL= PLAQUE_WITH_WALL,
            CATEGORIES=CATEGORIES,
            EPOCHS=EPOCHS

        )
    elif TYPE_OF_NETWORK == "maag_weak":    
        model, train_results = maag_run_segmentation_with_weak(
            DEVICE,
            TRAIN_IMG_PATH=TRAIN_IMG_PATH,
            TRAIN_LABELS_PATH=TRAIN_LABELS_PATH,
            TRAIN_SCRIBLES_IMG_PATH= WEAK_IMG_PATH,
            TRAIN_SCRIBLES_LABELS_PATH= WEAK_LABELS_PATH,
            VAL_IMG_PATH = VAL_IMG_PATH,
            VAL_LABELS_PATH = VAL_LABELS_PATH,
            TRAIN_TRANSFORMATIONS_SEG = TRAIN_TRANSFORMATIONS_SEG,
            VAL_TRANSFORMATIONS_SEG = VAL_TRANSFORMATIONS_SEG,
            TRANSFORMATIONS_TORCH =TRANSFORMATIONS_TORCH,
            WEAK_ANNOTATIONS_TRANSFORMATIONS_TORCH=WEAK_ANNOTATIONS_TRANSFORMATIONS_TORCH,
            PLAQUE_WITH_WALL= PLAQUE_WITH_WALL,
            CATEGORIES=CATEGORIES,
            EPOCHS=EPOCHS
        )
    elif TYPE_OF_NETWORK == "acnn_sim":
        model, train_results = accn_sim_run_training_segmentation(
            DEVICE,
            TRAIN_IMG_PATH,
            TRAIN_LABELS_PATH, 
            VAL_IMG_PATH,
            VAL_LABELS_PATH,
            TRAIN_TRANSFORMATIONS_SEG,
            VAL_TRANSFORMATIONS_SEG,
            TRANSFORMATIONS_TORCH,
            PLAQUE_WITH_WALL,
            CATEGORIES,
            EPOCHS, 
            EXPERIMENT_PATH
        )
    elif TYPE_OF_NETWORK == "acnn_seq":
        model, train_results = accn_seq_run_training_segmentation(
            DEVICE,
            TRAIN_IMG_PATH,
            TRAIN_LABELS_PATH, 
            VAL_IMG_PATH,
            VAL_LABELS_PATH,
            TRAIN_TRANSFORMATIONS_SEG,
            VAL_TRANSFORMATIONS_SEG,
            TRANSFORMATIONS_TORCH,
            PLAQUE_WITH_WALL,
            CATEGORIES,
            EPOCHS, 
            EXPERIMENT_PATH
        )

    else:
        print(f"ERROR: model of type {TYPE_OF_NETWORK} do not exists")
        sys.exit()


    torch.save(model.state_dict(), join(EXPERIMENT_PATH, f"{EXPERIMENT_NAME}.pt"))
    create_train_val_graphs(train_results, experiment_path= EXPERIMENT_PATH)
    with open(join(EXPERIMENT_PATH, "train_val_results.json"), "w") as fp:
        dump(train_results, fp)
    
    test_segmentor(
        DEVICE,
        model,
        EXPERIMENT_PATH,
        TEST_IMG_PATH,
        TEST_LABELS_PATH,
        TRANSFORMATIONS_TORCH,
        VAL_TRANSFORMATIONS_SEG,
        CATEGORIES,
        image_shape=MODEL_INPUT_SHAPE,
        type_of_architecture=TYPE_OF_NETWORK)

    

    

    

    
    

    