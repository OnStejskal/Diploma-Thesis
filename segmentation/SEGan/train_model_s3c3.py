from copy import deepcopy

from torch import device, logical_and, logical_or, no_grad, Tensor, mean
from torch import device, set_grad_enabled
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)
from SEGan.loss import dice_loss, dice_loss_multiclass
from SEGan.net import NetC, NetS
from common.metrics import classes_IoU, classses_dice_score
import numpy as np
import matplotlib.pyplot as plt
from common.dataset import SegmentationDataset
from common.visualization import plot_image, plot_segmentation, plot_image_label
from common.result_saver import ResultSaver
from common.transformations import (
    SegCompose,
    SegCrop,
    SegRandomHorizontalFlip,
    SegRandomVerticalFlip,
)

def segan_run_segmentation(
    DEVICE: device,
    TRAIN_IMG_PATH: str,
    TRAIN_LABELS_PATH: str,
    VAL_IMG_PATH: str,
    VAL_LABELS_PATH: str,
    TRAIN_TRANSFORMATIONS_SEG: SegCompose,
    VAL_TRANSFORMATIONS_SEG: SegCompose,
    TRANSFORMATIONS_TORCH: Compose,
    PLAQUE_WITH_WALL = False,
    CATEGORIES = 4,
    EPOCHS = 1
) -> tuple[Module, dict]:
    """Function that run the training of the segmentation network via the segan framerwork with 3 discrimantoe and 1 segmentor settings
    Args:
        DEVICE (device): device
        TRAIN_IMG_PATH (str): path to train images
        TRAIN_LABELS_PATH (str): path to train segmentations
        VAL_IMG_PATH (str): path to validation images
        VAL_LABELS_PATH (str): path to validation segmentataions
        TRAIN_TRANSFORMATIONS_SEG (SegCompose):
        VAL_TRANSFORMATIONS_SEG (SegCompose): transformations specific for segmentation
        TRANSFORMATIONS_TORCH (Compose): general pytorch transformations
        PLAQUE_WITH_WALL (bool, optional): Join plaque and wall into one class. Defaults to False.
        CATEGORIES (int, optional): Number of classes. Defaults to 4.
        EPOCHS (int, optional): number of epochs. Defaults to 1.

    Returns:
        tuple[Module, dict]: best segmentor model network and dictionary containing training and validation details
    """

    train_dataset = SegmentationDataset(
        TRAIN_IMG_PATH, TRAIN_LABELS_PATH, TRAIN_TRANSFORMATIONS_SEG, TRANSFORMATIONS_TORCH, plaque_with_wall=PLAQUE_WITH_WALL
    )

    val_dataset = SegmentationDataset(
        VAL_IMG_PATH, VAL_LABELS_PATH, VAL_TRANSFORMATIONS_SEG, TRANSFORMATIONS_TORCH, plaque_with_wall=PLAQUE_WITH_WALL
    )
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    number_of_classes_without_background = CATEGORIES - 1
    lr = 0.00002  
    decay = 0.5  # Learning rate decay
    alpha = 0.1 #0.1  # weight given to dice loss while generator training
    beta = 1. # weight given to the L1 loss while generator training
    clip_size = 0.05 # absolute value of clapping discriminator networks

    return train_models(
        train_loader,
        val_loader,
        DEVICE,
        EPOCHS,
        number_of_classes_without_background,
        alpha,
        beta,
        decay,
        lr,
        clip_size
    )

def train_models(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: device,
    num_epochs: int,
    n_classes: int,
    alpha: float,
    beta: float,
    decay: float,
    lr: float,
    clip_size: float,
) -> list[Module, dict]:
    """Trains the segmentation model following the SEGan framework, with 3 discrimantors and 1 segmentor settings
    Args:
        train_loader (DataLoader): train dataloader
        val_loader (DataLoader): validation dataloader
        device (device): device for execution CPU or CUDA device
        num_epochs (int): number of epochs
        n_classes (int): number of classes
        alpha (float): weight given to dice loss while generator training
        beta (float): weight given to L1 loss while generator training
        decay (float): weight decau for optimizer
        lr (float): learning rate for optimizer
        clip_size (float): absolute value of clipping minimal and maximal value of weight in discriminator network

    Returns:
        tuple[Module, dict]: best segmentor model network and dictionary containing training and validation details
    """
    extra_train_vars = ['train_disc_l1_loss_1', 'train_disc_l1_loss_2', 'train_disc_l1_loss_3',"train_disc_l1_loss_mean", 'train_gen_l1_loss', 'train_gen_dice_loss']
    extra_val_vars = ['val_l1_loss', 'val_dice_loss']
    rs = ResultSaver(extra_train_variables=extra_train_vars, extra_val_variables=extra_val_vars, apply_softmax=False)

    train_size = len(train_loader)
    val_size = len(val_loader)
    print(f'train size: {train_size}, val size: {val_size}')

    g_model = NetS(classes=n_classes+1)
    g_model.to(device)
    g_optimizer = RMSprop(g_model.parameters(), lr=lr)

    d_models = []
    d_optimizers = []
    for i in range(n_classes):
        d_model = NetC()
        d_model.to(device)
        d_optimizer = RMSprop(d_model.parameters(), lr=lr)
        d_models.append(d_model)
        d_optimizers.append(d_optimizer)


    best_model = None
    max_iou = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 12)

        ####################################################### TRAINING #######################################################
        g_model.train()
        train_mode(d_models)
        for image, target_all_class in train_loader:
            image = image.to(device) # [batch x 3 x 512 x 512]
            target_all_class = target_all_class.to(device) # [batch x 512 X 512]

            ##################################
            ### train Discriminator/Critic ###
            ##################################
            output_all_class = g_model(image)
            output_all_class = F.softmax(output_all_class, dim=1) # [batch, 4, w,h]
            output_all_class = output_all_class.detach() ### detach G from the network
            disc_l1_losses = [0 for _ in range(n_classes)]
            for i in range(n_classes):
                target = (target_all_class == i + 1).to(torch.int)
                target = target.to(device) # [batch x 512 X 512]
                output = output_all_class[:,i+1,:,:].unsqueeze(1)
                d_models[i].zero_grad()
                
                output_masked = image * output.expand_as(image)
                output_masked = output_masked.to(device)
                target_masked = image * target.unsqueeze(1).expand_as(image)
                target_masked = target_masked.to(device)

                output_D = d_models[i](output_masked)
                target_D = d_models[i](target_masked)
                loss_D = - torch.mean(torch.abs(output_D - target_D))
                loss_D.backward()
                d_optimizers[i].step()
                disc_l1_losses[i] = loss_D

                ## clip parameters in D
                for p in d_models[i].parameters():
                    p.data.clamp_(-clip_size, clip_size)
            train_disc_l1_loss_1 = disc_l1_losses[0]
            train_disc_l1_loss_2 = disc_l1_losses[1]
            train_disc_l1_loss_3 = disc_l1_losses[2]
            train_disc_l1_loss_mean = sum(disc_l1_losses)/len(disc_l1_losses)

            #################################
            ### train Generator/Segmentor ###
            #################################
            g_model.zero_grad()

            output_all_class = g_model(image)
            output_all_class = F.softmax(output_all_class, dim = 1)

            loss_dice = dice_loss(output_all_class,target_all_class, include_background=True)
            
            losses_G = []
            for i in range(n_classes):
                target = (target_all_class == i + 1).to(torch.int)
                target = target.to(device) # [batch x 512 X 512]
                output = output_all_class[:,i+1,:,:].unsqueeze(1)
                output_masked = image * output.expand_as(image)
                output_masked = output_masked.to(device)
                target_masked = image * target.unsqueeze(1).expand_as(image)
                target_masked = target_masked.to(device)
                output_G = d_models[i](output_masked)
                target_G = d_models[i](target_masked)
                losses_G.append(torch.mean(torch.abs(output_G - target_G)))
            loss_G = sum(losses_G)/len(losses_G)
            beta = 1
            loss_G_joint = beta * loss_G + alpha * loss_dice
            loss_G_joint.backward()
            g_optimizer.step()

            rs.train_step([loss_G_joint, train_disc_l1_loss_1,train_disc_l1_loss_2,train_disc_l1_loss_3, train_disc_l1_loss_mean, loss_G, loss_dice])

        ####################################################### VALIDATION #######################################################
        g_model.eval()
        eval_mode(d_models)
        with set_grad_enabled(False):
            for image, target_all_class in val_loader:
                image = image.to(device)
                target_all_class = target_all_class.to(device)
                output_all_class = g_model(image)
                output_all_class = F.softmax(output_all_class, dim=1)

                losses_G = []
                for i in range(n_classes):
                    target = (target_all_class == i + 1).to(torch.int)
                    target = target.to(device) # [batch x 512 X 512]
                    output = output_all_class[:,i+1,:,:].unsqueeze(1)
                    output_masked = image * output.expand_as(image)
                    output_masked = output_masked.to(device)
                    
                    target_masked = image * target.unsqueeze(1).expand_as(image)
                    target_masked = target_masked.to(device)
                    output_G = d_models[i](output_masked)
                    target_G = d_models[i](target_masked)
                    losses_G.append(torch.mean(torch.abs(output_G - target_G)))
                loss_G = sum(losses_G)/len(losses_G)
                loss_dice = dice_loss(output_all_class, target_all_class, apply_softamax=False)
                loss_joint = loss_G + alpha * loss_dice
                rs.val_step([loss_joint, loss_G, loss_dice], output_all_class, target_all_class)

        rs.epoch_step(print_all_score=True)

        #########################################################################################################################
        mIoU = rs.get_val_loss(custom_loss_name="val_ious_mean")
        if mIoU > max_iou:
            print("best model!")
            max_iou = mIoU
            best_model = deepcopy(g_model.state_dict())

        if epoch % 25 == 0:
            lr = lr*decay
            if lr <= 0.00000001:
                lr = 0.00000001
            print('Learning Rate: {:.6f}'.format(lr))
            print('Max mIoU: {:.4f}'.format(max_iou))
            g_optimizer = RMSprop(g_model.parameters(), lr=lr)
            for i in range(n_classes):
                d_optimizers[i] = RMSprop(d_models[i].parameters(), lr=lr)
    
    g_model.load_state_dict(best_model)
    return g_model, rs.results
 
def train_mode(models):
    for model in models:
        model.train()

def eval_mode(models):
    for model in models:
        model.eval()