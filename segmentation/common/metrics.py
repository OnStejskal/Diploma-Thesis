from torch import nn
import torch
from numpy import mean


# def mean(list):
#     sum(list)/len(list)

def mean_dice_score(predictions, labels, soft = True, include_background = True, apply_softamax = True):
    """return mean dice score over all classes 

    Args:
        predictions (tensor(batch, num_clases, height, width)): predictions from segmentation models 0 to 1 probabilites for class
        labels (tensor(batch, height, width)): labels for each pixel, values = 0...number of classes - 1
        soft (bool, optional): if True dice is computed as multiplication of probabilities. Defaults to True.
        include_background (bool, optional): if True include background class. Defaults to True.

    Returns:
        float: mean dice score
    """
    return mean(classses_dice_score(predictions, labels, soft, include_background, apply_softamax))

def classses_dice_score(predictions, labels, soft = True, include_background = True, apply_softamax = True):
    """compute dice score for every class as mean dice score over the batch

    Args:
        predictions (tensor(batch, num_clases, height, width)): predictions from segmentation models 0 to 1 probabilites for class
        labels (tensor(batch, height, width)): labels for each pixel, values = 0...number of classes - 1
        soft (bool, optional): if True dice is computed as multiplication of probabilities. Defaults to True.
        include_background (bool, optional): if True include background class. Defaults to True.

    Returns:
        list[float]: list of dice scores for each class (is computed as mmean from batch)
    """

    n_classes = predictions.shape[1]
    labels_oh = nn.functional.one_hot(labels, num_classes=n_classes).permute((0,3,1,2))
    if apply_softamax:
        predictions = nn.functional.softmax(predictions, dim = 1)

    if not soft:
        predictions = torch.argmax(predictions, dim=1)
        predictions = nn.functional.one_hot(predictions, num_classes = n_classes).permute((0,3,1,2))

    if not include_background:
        predictions = predictions[:,1:, :, :]
        labels_oh = labels_oh[:,1:,:,:]

    intersection = torch.sum(predictions * labels_oh, dim=(2,3))
    union = torch.sum(predictions, dim=(2,3)) + torch.sum(labels_oh, dim=(2,3))   
    dice = (2.0 * intersection) / (union)  # Adding a small epsilon to avoid division by zero
    return torch.mean(dice, dim=0).tolist()

def mean_IoU(predictions, labels, include_background = True):
    """compute mean iou over all classes

    Args:
        predictions (tensor(batch, num_clases, height, width)): predictions from segmentation models 0 to 1 probabilites for class
        labels (tensor(batch, height, width)): labels for each pixel, values = 0...number of classes - 1
        include_background (bool, optional): if True include background class. Defaults to True.

    Returns:
        float: mean IOU
    """
    return mean(classes_IoU(predictions, labels, include_background))


def classes_IoU(prediction, label, include_background = True):
    """compute iou for every class as mean iou over the batch

    Args:
        predictions (tensor(batch, num_clases, height, width)): predictions from segmentation models 0 to 1 probabilites for class
        labels (tensor(batch, height, width)): labels for each pixel, values = 0...number of classes - 1
        include_background (bool, optional): if True include background class. Defaults to True.

    Returns:
        float: mean IOU
    """
    epsilon = 1e-6
    n_classes = prediction.shape[1]
    batch_size = prediction.shape[0]
    prediction = prediction.argmax(1)
    prediction_oh = nn.functional.one_hot(prediction, num_classes = n_classes).permute((0,3,1,2))
    label_oh = nn.functional.one_hot(label, num_classes = n_classes).permute((0,3,1,2))
    # print(prediction_oh.shape)
    # print(label_oh.shape)

    if not include_background:
        n_classes -= 1
        prediction_oh = prediction_oh[:, 1:, :,:]
        label_oh = label_oh[:, 1:, :,:]

    classes_iou = [[] for _ in range(n_classes)]
    for b in range(batch_size):
        for i in range(n_classes):
            intersection = torch.sum(prediction_oh[b,i,:,:] * label_oh[b,i,:,:])
            union = torch.sum(prediction_oh[b,i,:,:]) + torch.sum(label_oh[b,i,:,:]) - intersection
            classes_iou[i].append(
                ((intersection + epsilon) / (union + epsilon)).tolist()
            )
    mean_classes_iou = [sum(class_iou)/batch_size for class_iou in classes_iou]
    return mean_classes_iou



def multiclass_dice_score_sumed_over_all_dimensions_including_class_dim(predictions, labels, soft = True, include_background = True, apply_softamax = True):
    """compute dice score for every class as mean dice score over the batch

    Args:
        predictions (tensor(batch, num_clases, height, width)): predictions from segmentation models 0 to 1 probabilites for class
        labels (tensor(batch, height, width)): labels for each pixel, values = 0...number of classes - 1
        soft (bool, optional): if True dice is computed as multiplication of probabilities. Defaults to True.
        include_background (bool, optional): if True include background class. Defaults to True.

    Returns:
        list[float]: list of dice scores for each class (is computed as mmean from batch)
    """

    n_classes = predictions.shape[1]
    labels_oh = nn.functional.one_hot(labels, num_classes=n_classes).permute((0,3,1,2))
    if apply_softamax:
        predictions = nn.functional.softmax(predictions, dim = 1)

    if not soft:
        predictions = torch.argmax(predictions, dim=1)
        predictions = nn.functional.one_hot(predictions, num_classes = n_classes).permute((0,3,1,2))

    if not include_background:
        predictions = predictions[:,1:, :, :]
        labels_oh = labels_oh[:,1:,:,:]

    intersection = torch.sum(predictions * labels_oh, dim=(1,2,3))
    union = torch.sum(predictions, dim=(1,2,3)) + torch.sum(labels_oh, dim=(1,2,3))   
    dice = (2.0 * intersection) / (union)  # Adding a small epsilon to avoid division by zero
    return torch.mean(dice, dim=0).item()


def binary_dice_score(output, target, soft = True, apply_sigmoid = False, threshold=0.5):
    """compute binary dice score for class x background probability

    Args:
        output (tensor(batch, 1, w, h)): _description_
        target (tensor(batch, w, h)): _description_
        threshold (float, optional): _description_. Defaults to 0.5.
        soft (bool, optional): if True dice is computed as multiplication of probabilities, if false argmax is aplied. Defaults to True.
        apply_sigmoid (bool, optional): if True apply sigmoid on the net output. Defaults to True.
    Returns:
        flaot: dice score
    """
    # Convert the tensors to binary masks
    output = output.squeeze(dim=1)
    print(output.shape)

    print(target.shape)
    if apply_sigmoid:
        output = torch.nn.functional.sigmoid(output)
    if not soft:
        output = (output > threshold).float()
        target = (target > threshold).float()
    # Calculate intersection and union

    intersection = torch.sum(output * target, dim=(1,2))
    union = torch.sum(output, dim=(1,2)) + torch.sum(target, dim=(1,2))
    print(intersection)
    print(union)
    # Avoid division by zero
    epsilon = 1e-8
    dice_score = (2 * intersection + epsilon) / (union + epsilon)
    mean_dice = torch.mean(dice_score)
    return mean_dice.item()


def binary_iou(output, target, apply_sigmoid = False, threshold=0.5):
    output = output.squeeze(dim=1)
    print(output.shape)

    print(target.shape)
    if apply_sigmoid:
        output = torch.nn.functional.sigmoid(output)
    output = (output > threshold).float()
    target = (target > threshold).float()
    # Calculate intersection and union
    intersection = torch.sum(output * target, dim=(1,2))
    union = torch.sum(output, dim=(1,2)) + torch.sum(target, dim=(1,2)) - intersection

    # Avoid division by zero
    epsilon = 1e-8
    iou = (intersection + epsilon) / (union + epsilon)
    print(iou)
    mean_iou = torch.mean(iou) 
    return mean_iou.item()



def accuracy(outputs, targets):
    size = targets.numel()
    outputs = outputs.argmax(dim=1)
    # targets = targets.argmax(dim=1)
    return (outputs == targets).sum() / size

