from copy import deepcopy

from torch import device, set_grad_enabled
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from common.result_saver import ResultSaver
from common.dataset import SegmentationDataset
from Unet.model import Unet, UnetDVCFS
from torch.optim import RMSprop
from common.losses import LogCoshDiceLoss

def unet_run_segmentation(
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

    ################# BASE SETTING #################
    # model = Unet(CATEGORIES)
    # model.to(DEVICE)

    # loss = CrossEntropyLoss()

    # lr = 0.00001
    # patience = 25
    # optimizer = RMSprop(model.parameters(), lr=lr, momentum=0.99)
    # scheduler = ReduceLROnPlateau(optimizer, patience=patience,min_lr=10e-7)

    ################# BEST UNET SETTING #################
    model = UnetDVCFS(CATEGORIES)
    model.to(DEVICE)
    loss_weights=[1.0, 1.5, 1.75, 1.0]
    loss = LogCoshDiceLoss(loss_weights)
    lr = 0.00001
    patience = 25
    optimizer = RMSprop(model.parameters(), lr=lr, momentum=0.99)
    scheduler = ReduceLROnPlateau(optimizer, patience=patience,min_lr=10e-7)

    #####################################################

    return train_model(
        model,
        train_loader,
        val_loader,
        loss,
        optimizer,
        DEVICE,
        scheduler,
        EPOCHS,
    )

def train_model(
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss: _Loss,
    optimizer: Optimizer,
    device: device,
    scheduler: _LRScheduler,
    num_epochs: int,
) -> Module:
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
    

    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    rs = ResultSaver(apply_softmax=False)

    best_val_loss = 10 ** 8
    best_model = None
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 12)

        train_epoch_loss = 0.0
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with set_grad_enabled(True):
                outputs = model(inputs)
                l = loss(outputs, labels)
                l.backward()
                optimizer.step()

                rs.train_step(l)
                train_epoch_loss += l.item() * inputs.size(0)

        scheduler.step(train_epoch_loss)

        val_epoch_loss = 0.0
        model.eval()

        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with set_grad_enabled(False):
                outputs = model(inputs)
                l = loss(outputs, labels)
                val_epoch_loss += l.item() * inputs.size(0)
                rs.val_step(l, outputs, labels)

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model = deepcopy(model.state_dict())

        
        print(
            f"Train loss: {train_epoch_loss/train_size}, Val. loss: {val_epoch_loss/val_size}"
        )
        rs.epoch_step()

    model.load_state_dict(best_model)
    return model, rs.results
