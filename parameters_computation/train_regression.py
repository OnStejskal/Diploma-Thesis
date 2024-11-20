import sys
import json
import argparse
from copy import deepcopy
from torch import cuda, device, save
from dataset import RegressionDataset, mask_all_segments,mask_plaque, cat_segmentation_with_image, img_and_plaque_mask, img_and_all_segments_mask, img_and_one_hot_segmentation
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)
from scipy.stats import pearsonr, spearmanr


import numpy as np
from os.path import join, exists
from os import makedirs
from model_regression import Resnet18RegressionMaskMiddle, Resnet18RegressionParams, Resnet18RegressionParamsCatMiddle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from visualization import plot_image_label,plot_segmentation,plot_image, create_train_val_graphs
from transformations import (
    SegCompose, 
    SegCrop,
    SegRandomRotation,
    SegRandomHorizontalFlip,
    SegRandomBrightness,
    SegRandomContrast,
    SegRandomGammaCorrection,
    SegRandomVerticalFlip)

from test_regression import test_regression

#############################################################
### This file is for the training of the tegression       ###
### network determining non-geometrical parameter         ###
### from the image                                        ###
#############################################################

##### fill these #####
EXPERIMENT_NAME = "fill the name of the experiment"
MASKING_FUCNTION = "fill the masking function"
DATASET_NAME = "fill the dataset with segmentation"
TRANSFORMATIONS_CUSTOM = "fill the transformations"
ECHOGENECITY = "bool specifying whether echogenicity or homogeneity"
AT_WHICH_LAYER_IS_CAT_OR_MASK = "specify the layer where the inner masking is performed for midle models"
MODEL = "initialize the model type"

##### examples #####
# TRANSFORMATIONS_CUSTOM_COMPLEX = SegCompose(
#     [
#     SegRandomRotation(),
#     SegRandomContrast(),
#     SegRandomGammaCorrection(),
#     SegRandomBrightness(),
#     SegRandomHorizontalFlip(),
#     ])
# TRANSFORMATIONS_CUSTOM_NO = SegCompose([])

# EXPERIMENT_NAME = "FINAL_width_cat_midle"
# MASKING_FUCNTION = img_and_one_hot_segmentation
# AT_WHICH_LAYER_IS_CAT_OR_MASK = 6
# DATASET_NAME = "MAAG_FIX"
# TRANSFORMATIONS_CUSTOM = TRANSFORMATIONS_CUSTOM_COMPLEX
# MODEL = Resnet18RegressionParamsCatMiddle(pretrained=True, input_img_dim=3,input_seg_dim = 4)

# model = Resnet18NonGeomParamsCatMiddle(pretrained=True, input_img_dim=input_dimension[0], input_seg_dim=input_dimension[1], echogenicity= echogenicity, concat_layer=AT_WHICH_LAYER_IS_CAT_OR_MASK)
# # model = Resnet34NonGeomParams(pretrained=True, input_dim=input_dimension, echogenicity=echogenicity)
# model = Resnet18NonGeomParamsMaskMiddle(pretrained=True, input_img_dim=input_dimension[0], input_seg_dim=input_dimension[1], echogenicity= echogenicity, concat_layer=AT_WHICH_LAYER_IS_CAT_OR_MASK)
# model.to(device)

# EXPERIMENT_NAME = "regression_rectangle_data_bestunet_cat"
# MASKING_FUCNTION = cat_segmentation_with_image
# AT_WHICH_LAYER_IS_CAT_OR_MASK = 6
# DATASET_NAME = "UNET_best_model_segmentation_trainvaltest"
# TRANSFORMATIONS_CUSTOM = TRANSFORMATIONS_CUSTOM_SIMPLE
# MODEL = Resnet18RegressionParams(pretrained=True, input_dim=7)

# EXPERIMENT_NAME = "regression_rectangle_data_bestunet_maskplaque"
# MASKING_FUCNTION = mask_plaque
# AT_WHICH_LAYER_IS_CAT_OR_MASK = 6
# DATASET_NAME = "UNET_best_model_segmentation_trainvaltest"
# TRANSFORMATIONS_CUSTOM = TRANSFORMATIONS_CUSTOM_SIMPLE
# MODEL = Resnet18RegressionParams(pretrained=True, input_dim=3)

######################################################################################
# EXPERIMENT_NAME = "FINAL_width_mask_plaque"
# MASKING_FUCNTION = mask_plaque
# AT_WHICH_LAYER_IS_CAT_OR_MASK = 6
# DATASET_NAME = "MAAG_FIX"
# TRANSFORMATIONS_CUSTOM = TRANSFORMATIONS_CUSTOM_COMPLEX
# MODEL = Resnet18RegressionParams(pretrained=True, input_dim=3)

# EXPERIMENT_NAME = "FINAL_width_cat_start"
# MASKING_FUCNTION = cat_segmentation_with_image
# AT_WHICH_LAYER_IS_CAT_OR_MASK = 6
# DATASET_NAME = "MAAG_FIX"
# TRANSFORMATIONS_CUSTOM = TRANSFORMATIONS_CUSTOM_COMPLEX
# MODEL = Resnet18RegressionParams(pretrained=True, input_dim=7)

# EXPERIMENT_NAME = "FINAL_width_mask_midle"
# MASKING_FUCNTION = img_and_plaque_mask
# AT_WHICH_LAYER_IS_CAT_OR_MASK = 6
# DATASET_NAME = "MAAG_FIX"
# TRANSFORMATIONS_CUSTOM = TRANSFORMATIONS_CUSTOM_COMPLEX
# MODEL = Resnet18RegressionMaskMiddle(pretrained=True, input_img_dim=3,input_seg_dim=1)


LABEL_NAME = "forecast_key_dataset.csv"
EXPERIMENT_PATH = join("models", "regression", EXPERIMENT_NAME)
NUM_EPOCHS = 30


TRANSFORMATIONS_TORCH = Compose(
    [
        Resize((256, 256)),
        ToTensor(),
    ])

def train_multi_input_model(
    device,
    TRAIN_IMAGES_PATH,
    TRAIN_SEGMENTATIONS_PATH, 
    VAL_IMAGES_PATH, 
    VAL_SEGMENTATIONS_PATH,
    TEST_IMAGES_PATH, 
    TEST_SEGMENTATIONS_PATH,
    LABELS_PATH,
    TRANSFORMATION_TORCH,
    TRANSFORMATION_CUSTOM,
    EXPERIMENT_PATH,
    NUM_EPOCHS,
    MODEL,
):
    model = MODEL
    model.to(device)
    num_epochs = NUM_EPOCHS
    masking_function = MASKING_FUCNTION
    if masking_function.__name__ in ['mask_plaque', 'mask_all_segments', 'cat_segmentation_with_image']:
        return_img_seg_separately = False
    else:
        return_img_seg_separately = True
    print(f'return_img_seg_separately: {return_img_seg_separately}')
    
    experiment_name = EXPERIMENT_NAME
    
    

    Learning_rate = 5 * 10**(-6)
    Weight_decay = 0.00005

    loss_function = MSELoss()
    train_dataset = RegressionDataset(TRAIN_IMAGES_PATH, TRAIN_SEGMENTATIONS_PATH, LABELS_PATH,masking_function, transformations_torch = TRANSFORMATION_TORCH, transformations_custom = TRANSFORMATION_CUSTOM)
    val_dataset = RegressionDataset(VAL_IMAGES_PATH, VAL_SEGMENTATIONS_PATH, LABELS_PATH,masking_function, transformations_torch = TRANSFORMATION_TORCH, transformations_custom = TRANSFORMATIONS_CUSTOM_NO)
    test_dataset = RegressionDataset(TEST_IMAGES_PATH, TEST_SEGMENTATIONS_PATH, LABELS_PATH,masking_function, transformations_torch = TRANSFORMATION_TORCH, transformations_custom = TRANSFORMATIONS_CUSTOM_NO)
    print(f"train dataset length: {len(train_dataset)}")   
    print(f"val dataset length: {len(val_dataset)}")    
    print(f"test dataset length: {len(test_dataset)}")    
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    # input_dimension = (train_dataset[0][0][0].shape[0], train_dataset[0][0][1].shape[0]) if return_img_seg_separately else train_dataset[0][0].shape[0]
    
    optimizer = Adam(model.parameters(), lr = Learning_rate, weight_decay=Weight_decay)
    scheduler = StepLR(
        optimizer,
        step_size= 10,
        gamma= 0.1,
    )

    training_loss_history = []
    validation_loss_history = []
    spearmans_val = []
    pearsons_val = []
    best_epoch = 0
    best_epoch_t_loss = 0
    best_epoch_v_loss = 0
    min_loss = np.inf
    max_pearson = 0
    best_model = None

    for epoch in range(num_epochs):
        # ------------------ Training ------------------
        model.train()
        print("-" * 30)
        print(f"Epoch {epoch}/{num_epochs}, lr = {optimizer.param_groups[0]['lr']}")
        epoch_training_loss = 0.0  # loss over whole epoch
        for images, labels, name in train_dataloader:
            if return_img_seg_separately:
                images = (images[0].float().to(device), images[1].float().to(device))
            else:
                images = images.to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            # print("img ", images.shape)
            # print("out ", outputs.shape)
            # print("lab ", labels.shape)
            loss = loss_function(outputs, labels)
            # print("loss ", loss.shape)
            loss.backward()
            optimizer.step()
            epoch_training_loss +=  loss.item()
        scheduler.step()
        avg_tloss = epoch_training_loss / len(train_dataloader)
        training_loss_history.append(avg_tloss)

        # ------------------ Validating ------------------
        model.eval()
        epoch_validation_loss = 0.0
        all_labels = []  # all labels from validation
        all_outputs = []  # all predictions from validation

        with torch.no_grad():
            for images, labels, name in val_dataloader:
                if return_img_seg_separately:
                    images = (images[0].float().to(device), images[1].float().to(device))
                else:
                    images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                # print(labels)
                # print(outputs)
                # print(outputs.shape)
                vall_loss = loss_function(outputs, labels)
                epoch_validation_loss += vall_loss.item()
                all_outputs.extend(outputs.detach().cpu().numpy())  # Assuming outputs is a tensor
                all_labels.extend(labels.cpu().numpy())  

        
        avg_vloss = epoch_validation_loss / len(val_dataloader)
        pearson_corr, _ = pearsonr(all_labels, all_outputs)
        spearman_corr, _ = spearmanr(all_labels, all_outputs)
        # print(pearson_corr)
        # print(spearman_corr)
        pearsons_val.append(pearson_corr[0])
        spearmans_val.append(spearman_corr)
        validation_loss_history.append(avg_vloss)
        
        print(
            f"LOSS train: {avg_tloss:.4f} validation: {avg_vloss:.4f}"
            )
        print(
            f"Pearson: {pearson_corr}, Spearman: {spearman_corr}"
            )

        if max_pearson < pearson_corr: #min_loss > avg_vloss:
            print("Saving Best Model")
            best_epoch = epoch
            best_epoch_t_loss = epoch_training_loss / len(train_dataloader)
            best_epoch_v_loss = epoch_validation_loss / len(val_dataloader)
            max_pearson = pearson_corr
            #min_loss = avg_vloss
            best_model = deepcopy(model.state_dict())

    
    info_dict = {
        "best_epoch": best_epoch,
        "training_loss": best_epoch_t_loss,
        "validation_loss": best_epoch_v_loss,
        "training_loss_history": training_loss_history,
        "validation_loss_history": validation_loss_history,
        "pearsons": pearsons_val,
        "spearmans": spearmans_val
    }

    model.load_state_dict(best_model)
    create_train_val_graphs(info_dict, EXPERIMENT_PATH)
    save(model.state_dict(), join(EXPERIMENT_PATH, f"{experiment_name}.pt"))
    with open(join(EXPERIMENT_PATH, "train_val_results.json"), 'w') as json_file:
            json.dump(info_dict, json_file)
    test_regression(
        device,
        model,
        EXPERIMENT_PATH,
        test_dataloader,
        return_img_seg_separately,
        loss_function
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

    train_img_path = join("data",DATASET_NAME,"train","images")
    train_seg_path = join("data",DATASET_NAME,"train","segmentations")
    val_img_path = join("data",DATASET_NAME,"val","images")
    val_seg_path = join("data",DATASET_NAME, "val","segmentations")
    test_img_path = join("data",DATASET_NAME,"test","images")
    test_seg_path = join("data",DATASET_NAME, "test","segmentations")
    # label_path = join("data", "forecast_key_dataset.csv")
    label_path = join("data", LABEL_NAME)

    print(f"Experiment name: {EXPERIMENT_NAME}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"mask fun: {MASKING_FUCNTION.__name__}")


    train_multi_input_model(
        DEVICE,
        train_img_path,
        train_seg_path,
        val_img_path,
        val_seg_path,
        test_img_path,
        test_seg_path,
        label_path,
        TRANSFORMATIONS_TORCH,
        TRANSFORMATIONS_CUSTOM,
        EXPERIMENT_PATH,
        NUM_EPOCHS=NUM_EPOCHS,
        MODEL = MODEL
    )
