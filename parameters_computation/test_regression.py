from torch import set_grad_enabled, argmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from os.path import join
import json
from scipy.stats import pearsonr, spearmanr

def test_regression(
        device,
        model,
        path_to_experiment,
        dataloader,
        return_img_seg_separately,
        loss_function, 
):
    """ test the regression model on the test dataset and create the results direcotry

    Args:
        device (device): cuda device
        model (nn.Model): pytorch model
        path_to_experiment (str): path to the experiment
        dataloader (dataloader): test dataloader
        return_img_seg_separately (bool): indicates whether the mask function returns separately images and segmentations
        loss_function (Loss): loss fucntion
    """
    model.eval()
    set_grad_enabled(False)
    loss_sum = 0
    all_outputs = []
    all_labels = []
    for images, labels, name in dataloader:
        if return_img_seg_separately:
            images = (images[0].float().to(device), images[1].float().to(device))
        else:
            images = images.to(device)
        labels = labels.to(device)
        outputs = model(images).squeeze()
        loss = loss_function(outputs, labels)
        loss_sum += loss.item()
        # for i in range(batch_size):
        all_outputs.extend(outputs.cpu().numpy())  # Appending individual outputs
        all_labels.extend(labels.cpu().numpy())
    all_outputs = [a.item() for a in all_outputs]
    all_labels = [a.item() for a in all_labels]
    avg_vloss = loss_sum / len(dataloader)
    pearson, _ = pearsonr(all_outputs, all_labels)
    spearman, _ = spearmanr(all_outputs, all_labels)
    result_dict = {
        "loss": avg_vloss,
        "pearson": pearson.item(),
        "spearman": spearman.item(),
        "labels": all_labels,
        "outputs": all_outputs,
    }

    with open(join(path_to_experiment, "test_results.json"), 'w') as json_file:
            json.dump(result_dict, json_file)

    