import torch

def disckriminator_lost(output, target):
    return  1 - torch.mean(torch.abs(output - target))

def binary_dice_loss(input,target):
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."

    num = input*target
    num = torch.sum(num,dim=3)
    num = torch.sum(num,dim=2)

    den1 = input*input
    den1 = torch.sum(den1,dim=3)
    den1 = torch.sum(den1,dim=2)

    den2 = target*target
    den2 = torch.sum(den2,dim=3)
    den2 = torch.sum(den2,dim=2)

    dice = 2*(num/(den1+den2))

    dice_total = 1 - torch.sum(dice)/dice.size(0) #divide by batchsize

    return dice_total

def dice_loss_multiclass(input_all_class,target_all_class):
    assert input_all_class.dim() == 4, "Input must be a 4D Tensor."

    dice_totals = []
    for i in range(input_all_class.size(1)):
        input = input_all_class[:,i,:,:].unsqueeze(1)
        target = (target_all_class == i + 1).to(torch.int).unsqueeze(1)


        num = input*target
        num = torch.sum(num,dim=3)
        num = torch.sum(num,dim=2)

        den1 = input*input
        den1 = torch.sum(den1,dim=3)
        den1 = torch.sum(den1,dim=2)

        den2 = target*target
        den2 = torch.sum(den2,dim=3)
        den2 = torch.sum(den2,dim=2)

        dice = 2*(num/(den1+den2))

        dice_totals.append(1 - torch.sum(dice)/dice.size(0))#divide by batchsize

    return sum(dice_totals)/len(dice_totals)

def dice_loss(predictions, labels, include_background = False, apply_softamax = False):
    """compute dice loss as a mean dice score across all classes

    Args:
        predictions (tensor(batch, num_clases, height, width)): predictions from segmentation models 0 to 1 probabilites for class
        labels (tensor(batch, height, width)): labels for each pixel, values = 0...number of classes - 1
        include_background (bool, optional): if True include background class. Defaults to False.

    Returns:
        tensor[1]: loss
    """

    n_classes = predictions.shape[1]
    labels_oh = torch.nn.functional.one_hot(labels, num_classes=n_classes).permute((0,3,1,2))
    if apply_softamax:
        predictions = torch.nn.functional.softmax(predictions, dim = 1)

    # if not soft:
    #     predictions = torch.argmax(predictions, dim=1)
    #     predictions = nn.functional.one_hot(predictions, num_classes = n_classes).permute((0,3,1,2))

    if not include_background:
        predictions = predictions[:,1:, :, :]
        labels_oh = labels_oh[:,1:,:,:]

    intersection = torch.sum(predictions * labels_oh, dim=(1,2,3))
    union = torch.sum(predictions, dim=(1,2,3)) + torch.sum(labels_oh, dim=(1,2,3))
    dice_score = (2.0 * intersection) / (union)  # Adding a small epsilon to avoid division by zero
    return 1 - torch.mean(dice_score, dim=0)