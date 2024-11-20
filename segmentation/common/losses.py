from copy import deepcopy

from torch import device, logical_and, logical_or, no_grad, Tensor, mean
from torch import device, set_grad_enabled
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from SEGan.net import NetC, NetS
import numpy as np
import matplotlib.pyplot as plt

import torch

from torch import cosh, log, ones, Tensor
from torch.nn import Module, Softmax
from torch.nn.functional import softmax, one_hot


class DiceLoss(Module):
    """Dice loss used for the segmentation tasks."""

    def __init__(self, weights: list = []):
        """Initializes a dice loss function.

        Parameters
        ----------
        weights : list
            Rescaling weights given to each class.
        """
        super(DiceLoss, self).__init__()
        self.soft_max = Softmax(dim=1)
        self.weights = weights

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        """Computes dice loss between the input and the target values.

        Parameters
        ----------
        outputs : Tensor
            Values predicted by the model.
        targets : Tensor
            The target values.

        Returns
        -------
        Tensor
            The dice loss between the input and the target values.
        """
        outputs = self.soft_max(outputs.float())

        for i, w in enumerate(self.weights):
            targets[:, i, :, :] = targets[:, i, :, :] * w

        outputs = outputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (outputs * targets).sum()
        dice = (2.0 * intersection) / (outputs.sum() + targets.sum())

        return 1 - dice


class LogCoshDiceLoss(Module):
    def __init__(self, weights: list = []):
        """Initializes a log-cosh dice loss function.

        weights : list
            Rescaling weights given to each class.
        """
        super(LogCoshDiceLoss, self).__init__()
        self.dice_loss = DiceLoss(weights)

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        """Computes log-cosh dice loss between the input and the target values.

        Parameters
        ----------
        outputs : Tensor
            Values predicted by the model.
        targets : Tensor
            The target values.

        Returns
        -------
        Tensor
            The log-cosh dice loss between the input and the target values.
        """
        targets = one_hot(targets).permute(0,3,1,2)
        dice_loss = self.dice_loss(outputs, targets)

        return log(cosh(dice_loss))


def logcosh_dice_loss(outputs: Tensor, targets: Tensor, weights: list = []) -> Tensor:
    """Computes log-cosh dice loss between the input and the target values.

    Parameters
    ----------
    outputs : Tensor
        Values predicted by the model.
    targets : Tensor
        The target values.
    weights: list
        Rescaling weights given to each class.
    Returns
    -------
    Tensor
        The log-cosh dice loss between the input and the target values.
    """
    outputs = softmax(outputs.float(), dim=1)

    for i, w in enumerate(weights):
        targets[:, i, :, :] = targets[:, i, :, :] * w

    outputs = outputs.view(-1)
    targets = targets.view(-1)

    intersection = (outputs * targets).sum()
    dice = (2.0 * intersection) / (outputs.sum() + targets.sum())

    return log(cosh(1 - dice))


def dice_score(predictions, labels):

    labels_oh = torch.nn.functional.one_hot(labels).permute((0,3,1,2))
    intersection = torch.sum(predictions * labels_oh, dim=(1,2,3))
    union = torch.sum(predictions, dim=(1,2,3)) + torch.sum(labels_oh, dim=(1,2,3))
    
    # Calculate Dice score
    dice = (2.0 * intersection) / (union)  # Adding a small epsilon to avoid division by zero
    return torch.mean(dice).item()




