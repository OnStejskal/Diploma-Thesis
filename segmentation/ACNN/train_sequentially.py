from os import makedirs
from os.path import join
from copy import deepcopy

from torch import device, set_grad_enabled
from torch.nn import Module, CrossEntropyLoss
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.nn.functional import one_hot, mse_loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Adam, RMSprop, Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ACNN.segmentor import Unet_Dropout, UnetDVCFS
from ACNN.segmentor_archive import Unet
from ACNN.deepClustering_autoencoder import DeepClusteringDecoder,  DeepClusteringEncoder
from ACNN.alexnet_autoencoder import AlexNetDecoder, AlexNetEncoder
from ACNN.denoise_autoencoder import DenoiseDecoder, DenoiseEncoder
from ACNN.utils.augmentations import swap_pixels_batch

from common.result_saver import ResultSaver
from common.metrics import accuracy
from common.visualization import (
    plot_ae_input_output_segmentations,
    plot_ae_input_output,
)
from common.transformations import SegCompose
from common.dataset import SegmentationDataset
from common.losses import LogCoshDiceLoss


def accn_seq_run_training_segmentation(
    DEVICE: device,
    TRAIN_IMG_PATH: str,
    TRAIN_LABELS_PATH: str,
    VAL_IMG_PATH: str,
    VAL_LABELS_PATH: str,
    TRAIN_TRANSFORMATIONS_SEG: SegCompose,
    VAL_TRANSFORMATIONS_SEG: SegCompose,
    TRANSFORMATIONS_TORCH: Compose,
    PLAQUE_WITH_WALL=False,
    CATEGORIES=4,
    EPOCHS=1,
    EXPERIMENT_PATH="experiment",
) -> tuple[Module, dict]:
    """Function that run the training of the segmentation network via the acnn framerwork where first the autoencoder is trained and the segmentor using the autoencoder loss

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
        EXPERIMENT_PATH (_type_, optional): output folder path. Defaults to "experiment".

    Returns:
        tuple[Module, dict]: best segmentor model network and dictionary containing training and validation details
    """
    LATENT_VECTOR_SIZE = 256
    encoder = DeepClusteringEncoder(latent_vector_size=LATENT_VECTOR_SIZE)
    decoder = DeepClusteringDecoder(latent_vector_size=LATENT_VECTOR_SIZE)
    # encoder = DenoiseEncoder(latent_vector_size=LATENT_VECTOR_SIZE)
    # decoder = DenoiseDecoder(latent_vector_size=LATENT_VECTOR_SIZE)
    # encoder = AlexNetEncoder(latent_vector_size=LATENT_VECTOR_SIZE)
    # decoder = AlexNetDecoder(latent_vector_size=LATENT_VECTOR_SIZE)
    encoder.to(DEVICE)
    decoder.to(DEVICE)

    ################# BASE SETTING #################
    lambda1 = 0.1  # weight of the MSE loss
    segmentor = Unet_Dropout(CATEGORIES)
    segmentor.to(DEVICE)

    lr = 1e-4  # Learning Rate
    beta1 = 0.5  # beta1 for adam
    momentum = 0.999
    weight_decay = 0.001
    autoencoder_optimizer = Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr,
        betas=(beta1, momentum),
    )
    segmentor_optimizer = Adam(
        segmentor.parameters(),
        lr=lr,
        betas=(beta1, momentum),
        weight_decay=weight_decay,
    )
    patience = 25
    min_lr = 10e-7  # minimum learning rate for scheduler
    scheduler_segmentor = ReduceLROnPlateau(
        segmentor_optimizer, patience=patience, min_lr=min_lr
    )
    scheduler_ae = ReduceLROnPlateau(
        autoencoder_optimizer, patience=patience, min_lr=min_lr
    )

    loss_function_seg = CrossEntropyLoss()
    loss_function_ae = CrossEntropyLoss()

    ################# BEST UNET SETTING #################
    # lambda1 = 0.1
    # segmentor = UnetDVCFS(CATEGORIES)
    # segmentor.to(DEVICE)

    # lr_ae = 0.0001
    # lr_seg = 0.00001
    # momentum = 0.99
    # autoencoder_optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr_ae, betas=(0.5, 0.999))
    # segmentor_optimizer = RMSprop(segmentor.parameters(), lr=lr_seg, momentum=momentum)

    # patience = 25
    # scheduler_segmentor = ReduceLROnPlateau(segmentor_optimizer, patience=patience,min_lr=10e-7)
    # scheduler_ae = ReduceLROnPlateau(autoencoder_optimizer, patience=patience,min_lr=10e-7)

    # loss_weights=[1.0, 1.5, 1.75, 1.0]
    # loss_function_seg = LogCoshDiceLoss(loss_weights)
    # loss_function_ae =  CrossEntropyLoss() #LogCoshDiceLoss(loss_weights)

    CREATE_AE_SEGMENTATIONS = (
        True  # denotes whether to plot the segmentations of the autoencoder
    )

    train_dataset = SegmentationDataset(
        TRAIN_IMG_PATH,
        TRAIN_LABELS_PATH,
        TRAIN_TRANSFORMATIONS_SEG,
        TRANSFORMATIONS_TORCH,
        plaque_with_wall=PLAQUE_WITH_WALL,
    )
    val_dataset = SegmentationDataset(
        VAL_IMG_PATH,
        VAL_LABELS_PATH,
        VAL_TRANSFORMATIONS_SEG,
        TRANSFORMATIONS_TORCH,
        plaque_with_wall=PLAQUE_WITH_WALL,
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    return train_model(
        segmentor,
        encoder,
        decoder,
        segmentor_optimizer,
        autoencoder_optimizer,
        scheduler_segmentor,
        scheduler_ae,
        loss_function_seg,
        loss_function_ae,
        train_loader,
        val_loader,
        DEVICE,
        EPOCHS,
        lambda1,
        EXPERIMENT_PATH,
        CREATE_AE_SEGMENTATIONS,
    )


def train_model(
    segmentor: Module,
    encoder: Module,
    decoder: Module,
    segmentor_optimizer: Optimizer,
    autoencoder_optimizer: Optimizer,
    scheduler_segmentor: _LRScheduler,
    scheduler_ae: _LRScheduler,
    loss_function_seg: _Loss,
    loss_function_ae: _Loss,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: device,
    num_epochs: int,
    lambda1: float,
    EXPERIMENT_PATH: str,
    CREATE_AE_SEGMENTATIONS: bool,
) -> tuple[Module, dict]:
    """Trains the segmentation model following the ACNN framework, first the autoencoder is trained then the segmentor is trained

    Args:
        segmentor (Module): segmentation network
        encoder (Module): encoder network
        decoder (Module): decoder network
        segmentor_optimizer (Optimizer): optimizer for segmentor network
        autoencoder_optimizer (Optimizer): optimizer for autoencoder network
        scheduler_segmentor (_LRScheduler): secheduler for segmentor training
        scheduler_ae (_LRScheduler): scheduler for autoencoder training
        loss_function_seg (_Loss): loss function for segmentor
        loss_function_ae (_Loss): loss function for autoencoder
        train_loader (DataLoader): train dataloader
        val_loader (DataLoader): validation dataloader
        device (device): device for execution CPU or CUDA device
        num_epochs (int): number of epochs
        lambda1 (float): coefficient which the MSE loss is multiplied
        EXPERIMENT_PATH (str): output folder pat
        CREATE_AE_SEGMENTATIONS (bool): if true autoencoder input and output are visualized

    Returns:
        tuple[Module, dict]: best segmentor model network and dictionary containing training and validation details
    """

    ####################################################### TRAINING AUTOENCODER #######################################################
    print("TRAINING AUTOENCODER")

    best_ae_loss = 10e7
    ae_train_losses = []
    ae_val_losses = []
    ae_accuracies = []
    best_encoder = None
    best_decoder = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 12)

        ######################## TRAIN PART ########################
        train_epoch_ae_loss = 0.0
        encoder.train()
        decoder.train()
        for _, label in train_loader:
            label = label.to(device)
            autoencoder_optimizer.zero_grad()

            encoded_label = encoder(one_hot(label).permute((0, 3, 1, 2)).float())
            decoded_label = decoder(encoded_label)

            label_swapped = swap_pixels_batch(label.clone())

            autoencoder_loss = loss_function_ae(decoded_label, label_swapped)
            autoencoder_loss.backward()
            autoencoder_optimizer.step()
            train_epoch_ae_loss += autoencoder_loss.item()
        scheduler_ae.step(train_epoch_ae_loss)
        ae_train_losses.append(train_epoch_ae_loss / len(train_loader))

        ######################## VALIDATION PART ########################
        val_epoch_ae_loss = 0.0
        val_epoch_accurancy = 0.0
        encoder.eval()
        decoder.eval()
        for i, (_, label) in enumerate(val_loader):
            label = label.to(device)
            with set_grad_enabled(False):
                oh_label = one_hot(label).permute((0, 3, 1, 2)).float()
                encoded_label = encoder(oh_label)
                decoded_label = decoder(encoded_label)
                vloss = loss_function_ae(decoded_label, label)
                val_epoch_ae_loss += vloss.item()
                val_epoch_accurancy += accuracy(decoded_label, label).item()

                # Every 50th episode some input and output segmentations are saved
                if ((epoch % 50 == 0) or (epoch == num_epochs - 1)) and i < 10:
                    current_epoch_path = join(
                        EXPERIMENT_PATH, "ae_traning", f"epoch_{epoch}"
                    )
                    makedirs(current_epoch_path, exist_ok=True)
                    plot_ae_input_output(
                        in_segmentation=oh_label[0].cpu().detach().numpy().argmax(0),
                        out_segmentation=decoded_label[0]
                        .cpu()
                        .detach()
                        .numpy()
                        .argmax(0),
                        img_name=str(i),
                        save_path=current_epoch_path,
                    )

        ae_accuracies.append(val_epoch_accurancy / len(val_loader))
        ae_val_losses.append(val_epoch_ae_loss / len(val_loader))
        ###############################################################

        print(f"Autoencoder Train loss: {train_epoch_ae_loss/len(train_loader)}")
        print(f"Autoencoder Validation loss: {val_epoch_ae_loss/len(val_loader)}")

        if val_epoch_ae_loss < best_ae_loss:
            best_ae_loss = val_epoch_ae_loss
            best_encoder = encoder.state_dict()
            best_decoder = decoder.state_dict()

    encoder.load_state_dict(best_encoder)
    decoder.load_state_dict(best_decoder)
    encoder.eval()

    ####################################################### TRAINING SEGMENTOR #######################################################
    print("TRAINING Segmentor")
    best_model = None
    best_val_loss = 10e7
    train_keys = ["train_seg_loss_sup", "train_seg_loss_euclid"]
    val_keys = ["val_seg_loss_sup", "val_euclid_loss"]
    rs = ResultSaver(train_keys, val_keys)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 12)

        ######################## TRAIN PART ########################
        segmentor.train()
        train_epoch_loss = 0.0

        for input, label in train_loader:
            segmentor_optimizer.zero_grad()

            input = input.to(device)
            label = label.to(device)

            segmentation = segmentor(input)
            encoded_seg = encoder(segmentation)
            encoded_label = encoder(one_hot(label).permute((0, 3, 1, 2)).float())

            segmentor_sup_loss = loss_function_seg(segmentation, label)
            euclid_loss = mse_loss(encoded_seg, encoded_label)

            # weighting the euclid loss because the MSE loss was too high in the experiments
            w_euclid = segmentor_sup_loss / euclid_loss
            segmentor_loss = segmentor_sup_loss + w_euclid * lambda1 * euclid_loss
            train_epoch_loss += segmentor_loss.item()
            segmentor_loss.backward()
            segmentor_optimizer.step()
            rs.train_step([segmentor_loss, segmentor_sup_loss, euclid_loss])

        scheduler_segmentor.step(train_epoch_loss)

        ######################## VALIDATION PART ########################
        segmentor.eval()

        for i, (input, label) in enumerate(val_loader):
            input = input.to(device)
            label = label.to(device)

            with set_grad_enabled(False):
                output = segmentor(input)
                seg_sup_loss = loss_function_seg(output, label)

                encoded_segmentation = encoder(output)
                decoded_segmentation = decoder(encoded_segmentation)
                encoded_label = encoder(one_hot(label).permute((0, 3, 1, 2)).float())

                euclid_loss = mse_loss(encoded_segmentation, encoded_label)

                w_euclid = seg_sup_loss / euclid_loss
                seg_loss = seg_sup_loss + w_euclid * lambda1 * euclid_loss
                rs.val_step([seg_loss, seg_sup_loss, euclid_loss], output, label)

                # Every 50th episode some segmentor outputs, autoencoder outputs and labels are saved for visualization
                if (
                    ((epoch % 50 == 0) or (epoch == num_epochs - 1))
                    and i < 10
                    and CREATE_AE_SEGMENTATIONS
                ):
                    current_epoch_path = join(
                        EXPERIMENT_PATH, "val_ae_segmentations", f"epoch_{epoch}"
                    )
                    makedirs(current_epoch_path, exist_ok=True)
                    plot_ae_input_output_segmentations(
                        in_segmentation=output[0].cpu().detach().numpy().argmax(0),
                        out_segmentation=decoded_segmentation[0]
                        .cpu()
                        .detach()
                        .numpy()
                        .argmax(0),
                        label=label[0].cpu().detach().numpy(),
                        img_name=str(i),
                        save_path=current_epoch_path,
                    )

        ###############################################################
        rs.epoch_step(print_all_score=True)

        # model with lowest validation loss is saved
        val_loss = rs.get_val_loss()
        if val_loss < best_val_loss:
            print("saving")
            best_val_loss = val_loss
            best_model = deepcopy(segmentor.state_dict())

    segmentor.load_state_dict(best_model)
    rs.results["ae_val_losses"] = ae_val_losses
    rs.results["ae_accuracies"] = ae_accuracies
    rs.results["ae_train_losses"] = ae_train_losses

    return segmentor, rs.results
