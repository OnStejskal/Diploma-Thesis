from copy import deepcopy

from torch import device, set_grad_enabled, tensor, mean, abs
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.nn.functional import one_hot, interpolate
from torch.optim.lr_scheduler import _LRScheduler, CyclicLR
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from MAAG.losses import wce_already_sofmaxed
from MAAG.architectures.discriminator import MultiResDiscriminator
from MAAG.architectures.segmentor_UNET import Segmentor_with_AG

from common.result_saver import ResultSaver
from common.dataset import SegmentationDataset
from common.transformations import (
    SegCompose,
)


RETURN_ATTENTIONS = True

def maag_run_segmentation(
    DEVICE: device,
    TRAIN_IMG_PATH: str,
    TRAIN_LABELS_PATH: str,
    VAL_IMG_PATH: str,
    VAL_LABELS_PATH: str,
    TRAIN_TRANSFORMATIONS_SEG: SegCompose,
    VAL_TRANSFORMATIONS_SEG: SegCompose,
    TRANSFORMATIONS_TORCH: SegCompose,
    PLAQUE_WITH_WALL = False,
    CATEGORIES = 4,
    EPOCHS = 1
) -> tuple[Module, dict]:
    """Function that run the training of the segmentation network via the maag framerwork.
    Joint training of the Segmentor and discriminator

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

    batch_size = 2
    train_loader_paired = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader_only_images = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader_only_labels = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    g_model = Segmentor_with_AG(input_channels=3, return_attentions=RETURN_ATTENTIONS)
    d_model = MultiResDiscriminator()

    g_model.to(DEVICE)
    d_model.to(DEVICE)

    lr = 1e-4  # Learning Rate
    betas = (0.5, 0.999) # betas for adam
    g_optimizer = Adam(g_model.parameters(), lr=lr, betas=betas)
    d_optimizer = Adam(d_model.parameters(), lr=lr, betas=betas)

    scheduler = CyclicLR(g_optimizer, base_lr=0.00001, max_lr=0.0001, step_size_up=20, step_size_down=20, cycle_momentum=False)

    a1 = 0.1
    a2 = 0.2
    a3 = 0.

    return train_model(
        train_loader_paired,
        train_loader_only_images,
        train_loader_only_labels,
        val_loader,
        g_model,
        d_model,
        g_optimizer,
        d_optimizer,
        scheduler,
        DEVICE,
        EPOCHS,
        a1,
        a2,
        a3  
    )


def train_model(
    train_loader_paired: DataLoader,
    train_loader_only_images: DataLoader,
    train_loader_only_labels: DataLoader,
    val_loader: DataLoader,
    g_model: Module,
    d_model: Module,
    g_optimizer: Optimizer,
    d_optimizer: Optimizer,
    scheduler: _LRScheduler,
    device: device,
    num_epochs: int,
    a1 = 0.1,
    a2 = 0.2,
    a3 = 0.3,
) -> list[Module, dict]:   
    """rains the segmentation model following the Valvano 2021 MAAG framework. This is the version of training only with fully annotated labels

    Args:
        train_loader_paired (DataLoader): dataloader with paired image with segmentations
        train_loader_only_images (DataLoader): dataloader that is used only for images
        train_loader_only_labels (DataLoader): dataloader that is used only for segmntations
        val_loader (DataLoader): dataloader with paired image with segmentations for validation
        g_model (Module): segmentor model
        d_model (Module): discriminator model
        g_optimizer (Optimizer): optimizer for segmentor
        d_optimizer (Optimizer): optimizer for discriminator
        scheduler (_LRScheduler): scheduler for generator optimizer
        device (device): device for execution CPU or CUDA device
        num_epochs (int): number of epochs
        a1 (float): weighting factor for generator adversarial loss in supervised loss function
        a2 (float): weighting factor for discriminator adversarial loss in unsupervised part
        a2 (float): weighting factor for generator adversarial loss in unsupervised part

    Returns:
        list[Module, dict]: best segmentor model network and dictionary containing training and validation details
    """

    best_model = None
    best_dice = 0.0
    extra_train_losses = ["train_sup_losses", "train_sup_wce_losses", "train_sup_disc_loss", "train_unsup_gen_loss", "train_unsup_disc_loss"]
    rs = ResultSaver(extra_train_losses, apply_softmax=False)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 12)
        
        g_model.train()
        d_model.train()
        ##################################################### TRAINING #####################################################
        for (sup_input, sup_label), (unsup_input, _), (_, disc_label) in zip(train_loader_paired, train_loader_only_images, train_loader_only_labels):    
            sup_input = sup_input.to(device)
            sup_label = sup_label.to(device)
            
            unsup_input = unsup_input.to(device)
            disc_label = disc_label.to(device)


            ################################### SUPERVISED STEP ###################################
            g_optimizer.zero_grad()
            if (RETURN_ATTENTIONS):
                sup_output_attentions = g_model(sup_input)
                sup_output = sup_output_attentions[0]
            else:
                sup_output = g_model(sup_input) # -> (batch, 4, w,h)
                sup_output_attentions = [sup_output] + g_model.attentions
            sup_wxe_loss = wce_already_sofmaxed(sup_output, sup_label)
            fake_disc_output = d_model(sup_output_attentions)

            sup_gen_adv_loss = 0.5 * mean((fake_disc_output - 1)**2)
            w_dynamic = abs(sup_wxe_loss / sup_gen_adv_loss)
            sup_loss = sup_wxe_loss + w_dynamic * a1 * sup_gen_adv_loss
            sup_loss.backward()
            g_optimizer.step()
            

            ################################### ADVERSARIAL STEP ###################################
            d_optimizer.zero_grad()
            if (RETURN_ATTENTIONS):
                unsup_output_attentions = g_model(unsup_input)
            else:
                unsup_output = g_model(unsup_input)
                unsup_output_attentions = [unsup_output] + g_model.attentions

            ##### OPTIMIZE DISCRIMINATOR #####
            real_attentions = create_real_attentions(disc_label)
            real_attentions = tuple(tensor.to(device) for tensor in real_attentions)
            real_disc_output = d_model(real_attentions)
            unsup_output_attentions_detached = [unsup_output_attention.detach() for unsup_output_attention in unsup_output_attentions]
            fake_disc_output = d_model(unsup_output_attentions_detached)
            unsup_disc_loss = 0.5 * mean((real_disc_output - 1.0) ** 2) + 0.5 * mean((fake_disc_output + 1.0) ** 2)
            unsup_disc_loss = a2 * unsup_disc_loss
            unsup_disc_loss.backward()
            d_optimizer.step()

            ##### OPTIMIZE GENERATOR #####
            g_optimizer.zero_grad()
            fake_disc_output = d_model(unsup_output_attentions)
            unsup_gen_loss = 0.5 * mean((fake_disc_output - 1)**2)
            unsup_gen_loss = a3 * unsup_gen_loss
            unsup_gen_loss.backward()
            g_optimizer.step()
            scheduler.step()

            
            rs.train_step([sup_loss,sup_loss, sup_wxe_loss, sup_gen_adv_loss, unsup_gen_loss, unsup_disc_loss])
        
        ##################################################### VALIDATION #####################################################
        g_model.eval()
        d_model.eval()
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with set_grad_enabled(False):
                output = g_model(inputs)
                if (RETURN_ATTENTIONS):
                    output = output[0]
                val_ce_loss = wce_already_sofmaxed(output, labels)
                rs.val_step(val_ce_loss,output,labels)

        rs.epoch_step(print_all_score=True)
        dice_score = rs.get_val_loss(custom_loss_name="val_dices_mean")

        # select based on best dice loss
        if  dice_score > best_dice:
            best_dice = dice_score
            print("best model!")
            best_model = deepcopy(g_model.state_dict())

    g_model.load_state_dict(best_model)
    return g_model, rs.results


def create_real_attentions(input: tensor) -> list[tensor]:
    """create attention tensors from real segmentation by downscaling

    Args:
        input (tensor): segmentation label

    Returns:
        list[tensor]: 4 attentions tensors
    """
    b, w, h = input.shape
    input_oh = one_hot(input).permute((0,3,1,2)).float()
    attention3 = interpolate(input_oh, size = (w//2, h //2), mode='bilinear', align_corners=False)
    attention2 = interpolate(input_oh, size = (w//4, h //4), mode='bilinear', align_corners=False)
    attention1 = interpolate(input_oh, size = (w//8, h //8), mode='bilinear', align_corners=False)
    return input_oh, attention3, attention2, attention1