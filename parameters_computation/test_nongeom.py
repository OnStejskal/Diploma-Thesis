from torch import set_grad_enabled, argmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from os.path import join
import json

def test_nongeom(
        device,
        model,
        path_to_experiment,
        dataloader,
        return_img_seg_separately,
        loss_function, 
):
    """ test the classifiacation model on the test dataset and create the results direcotry

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
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss_sum += loss.item()
        output_classes = argmax(outputs.detach(), dim=1)
        # for i in range(batch_size):
        all_outputs.extend(output_classes.cpu().numpy())  # Appending individual outputs
        all_labels.extend(labels.cpu().numpy())

    avg_vloss = loss_sum / len(dataloader)
    # all_outputs = np.concatenate(all_outputs, axis=0)
    # all_labels = np.concatenate(all_labels, axis=0)
    conf_matrix = confusion_matrix(all_labels, all_outputs)
    accuracy = accuracy_score(all_labels, all_outputs)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_outputs, average='weighted')
    
    result_dict = {
        "loss": avg_vloss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "conf_matrix": conf_matrix.tolist()
    }

    with open(join(path_to_experiment, "test_results.json"), 'w') as json_file:
            json.dump(result_dict, json_file)

    