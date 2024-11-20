from copy import deepcopy
import torch
from torch import device, set_grad_enabled, log, tensor, mean
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.nn.functional import one_hot, cross_entropy, mse_loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Resize

from ACNN.segmentor import Unet
from ACNN.alexnet_autoencoder import AlexNetDecoder, AlexNetEncoder
from ACNN.deepClustering_autoencoder import DeepClusteringDecoder, DeepClusteringEncoder
from ACNN.denoise_autoencoder import DenoiseDecoder, DenoiseEncoder


from common.result_saver import ResultSaver
from common.metrics import classses_dice_score
from common.dataset import SegmentationDataset

def accn_noae_run_training_segmentation(
    DEVICE,
    TRAIN_IMG_PATH: str,
    TRAIN_LABELS_PATH: str,
    VAL_IMG_PATH: str,
    VAL_LABELS_PATH: str,
    TRAIN_TRANSFORMATIONS_SEG,
    VAL_TRANSFORMATIONS_SEG,
    TRANSFORMATIONS_TORCH,
    PLAQUE_WITH_WALL = False,
    CATEGORIES = 4,
    EPOCHS = 1
):

    train_dataset = SegmentationDataset(
        TRAIN_IMG_PATH, TRAIN_LABELS_PATH, TRAIN_TRANSFORMATIONS_SEG, TRANSFORMATIONS_TORCH, plaque_with_wall=PLAQUE_WITH_WALL
    )

    val_dataset = SegmentationDataset(
        VAL_IMG_PATH, VAL_LABELS_PATH, VAL_TRANSFORMATIONS_SEG, TRANSFORMATIONS_TORCH, plaque_with_wall=PLAQUE_WITH_WALL
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    lr = 1e-4  # Learning Rate. Default=0.0002
    beta1 = 0.5  # beta1 for adam
    decay = 0.5  # Learning rate decay
    alpha = 0.1  # weight given to dice loss while generator training
    lambda1 = 0.1
    return train_model(
        train_loader,
        val_loader,
        DEVICE,
        EPOCHS,
        CATEGORIES,
        alpha,
        beta1,
        decay,
        lr,
        lambda1)

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: device,
    num_epochs: int,
    n_classes: int,
    alpha: float,
    beta1:float,
    decay: float,
    lr: float,
    lambda1: float, 

    
) -> list[Module]:
    """Trains the segmentation model on the training data.

    Parameters
    ----------
    model : Module
        Model to train.
    train_loader : DataLoader
        Train data.
    val_loader : DataLoader
        Train data.
    loss : _Loss
        Loss function.
    optimizer : Optimizer
        Selected optimizer which updates weights of the model
    device : device
        Device on which is the model.
    scheduler : Union[None, _LRScheduler]
        Selected scheduler of the learning rate.
    val_split : float
        Ratio of the train-validation split.
    num_epochs : int
        Number of training epochs.

    Returns
    -------
    tuple
        Model with best validation loss during the training.
    """

    lambda1 = 0.1

    segmentor = Unet(n_classes)
    # encoder = AlexNetEncoder()
    # decoder = AlexNetDecoder()
    

    segmentor.to(device)
    segmentor_optimizer = Adam(segmentor.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=0.0001)

    best_model = None

    train_keys = ["train_seg_loss_ce"]
    rs = ResultSaver(train_keys)
    
    best_val_loss = 10^6

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 12)
        segmentor.train()
        
        for input, label in train_loader: 

            input = input.to(device)
            label = label.to(device)

            ########## SEGMENTATION TRAIN #############
            segmentor_optimizer.zero_grad()
            segmentation = segmentor(input)
        
            segmentor_ce_loss = cross_entropy(segmentation, label)
            segmentor_loss = segmentor_ce_loss #+ lambda1*euclid_loss
            segmentor_loss.backward()
            segmentor_optimizer.step()
            rs.train_step([segmentor_loss, segmentor_ce_loss])

        segmentor.eval()
        
        for input, label in val_loader:
            input = input.to(device)
            label = label.to(device)

            with set_grad_enabled(False):
                output = segmentor(input)                
                ce_loss = cross_entropy(output, label)
                rs.val_step(ce_loss, output, label)


        rs.epoch_step(print_all_score=True)
        # train_keys = ["train_ae_loss", "train_ae_loss_euclid", "train_ae_loss_ce", "train_seg_loss_ce", "train_seg_loss_euclid"]
        # rs.print_last_value("train_ae_loss")
        # rs.print_last_value("train_ae_loss_ce")
        # rs.print_last_value("train_ae_loss_euclid")
        val_loss = rs.get_val_loss()
        if  val_loss < best_val_loss:
            print("saving")
            best_val_loss = val_loss
            best_model = deepcopy(segmentor.state_dict())


    segmentor.load_state_dict(best_model)
    return segmentor, rs.results
