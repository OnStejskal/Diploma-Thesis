import torch
import torch.nn.functional as F

def wce_already_sofmaxed(outputs, targets, num_classes = 4):
    """
    Weighted cross entropy loss for already softmaxed outputs.

    :param outputs: Softmax probabilities from the neural network,
                    shape [batch_size, num_classes, height, width].
    :param targets: Ground truth labels, shape [batch_size, height, width].
    :param class_weights: Tensor of shape [num_classes] with class weights.
    :return: Weighted cross entropy loss.
    """
    # Convert softmax outputs to log probabilities
    class_weights = []
    total_pixels = float(targets.numel())  # Total number of labeled pixels
    for c in range(0, num_classes):  # Start from 1 to ignore class 0
        class_pixels = float((targets == c).sum())
        class_weight = 1 - (class_pixels / total_pixels)
        class_weights.append(class_weight)
    class_weights = torch.tensor(class_weights).to(outputs.device)
    log_probs = torch.log(outputs + 1e-6)  # Adding epsilon for numerical stability
    loss = F.nll_loss(log_probs, targets, weight=class_weights, reduction='mean')
    return loss

def pwce_already_sofmaxed(outputs, targets, num_classes = 4):
    """
    Partial Weighted NLL loss for softmaxed outputs, computed only on annotated pixels.

    :param outputs: Softmax probabilities from the neural network,
                    shape [batch_size, num_classes, height, width].
    :param targets: Ground truth labels, shape [batch_size, height, width].
                    Annotated pixels are non-zero.
    :param class_weights: Tensor of shape [num_classes] with class weights.
    :return: Weighted NLL loss.
    """
    mask = targets != 0

    class_weights = [0]
    total_pixels = float(mask.sum())  # Total number of labeled pixels
    for c in range(1, num_classes):  # Start from 1 to ignore class 0
        class_pixels = float((targets == c).sum())
        class_weight = 1 - (class_pixels / total_pixels)
        class_weights.append(class_weight)
    class_weights = torch.tensor(class_weights).to(outputs.device)

    log_probs = torch.log(outputs + 1e-6)  # Add epsilon for numerical stability

    loss = F.nll_loss(log_probs, targets, weight=class_weights, reduction='mean')
    return loss