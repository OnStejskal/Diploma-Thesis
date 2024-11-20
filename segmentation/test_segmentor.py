from common.dataset import SegmentationEvaluationDataset
from torch.nn.functional import cross_entropy
from common.metrics import classes_IoU, classses_dice_score, accuracy
from torch import set_grad_enabled
from sklearn.metrics import confusion_matrix
import numpy as np
from json import dump
from os.path import join
from os import makedirs
from common.visualization import plot_segmentation_prediction_differences, plot_segmentation_prediction
from torch.nn.functional import one_hot

def test_segmentor(
        device,
        model,
        path_to_experiment,
        path_to_images,
        path_to_segmentations,
        transformation_torch, 
        transformation_seg,
        num_class = 4,
        image_shape: tuple = (256, 256),
        type_of_architecture= "unet"
        ):
    """ function that test the created segmentation is used in train_segmentation file after training and creates a test results

    Args:
        device (device): cuda device
        model (nn.Model): pytorch trained model
        path_to_experiment (str): path to the experiment
        path_to_images (str): path to the test set images
        path_to_segmentations (str): path to the test set segmentations
        transformation_torch (transformation): torchvision transformations
        transformation_seg (_type_): transforamtions spcicified in /common/transformation
        num_class (int, optional): numbr of segmented classes. Defaults to 4.
        image_shape (tuple, optional): shape of the fixed image if not specified it works as variable mode. Defaults to (256, 256).
        type_of_architecture (str, optional): type of the architecture. Defaults to "unet".
    """
    
    test_dataset = SegmentationEvaluationDataset(
        path_to_images,
        transformation_torch,
        transformation_seg,
        path_to_segmentations
    )

    makedirs(join(path_to_experiment,"segmentation_prediction"), exist_ok=True)
    makedirs(join(path_to_experiment,"segmentation_prediction_differences"), exist_ok=True)

    
    model.to(device)
    model.eval()
    losses = []
    accuracies = []
    mean_ious = []
    mean_dices = []
    ious = [[] for _ in range(num_class)]
    dices = [[] for _ in range(num_class)]
    cnf_matrices = []

    set_grad_enabled(False)
    for input, input_raw, img_name, label, label_raw in test_dataset:
        label_oh = one_hot(label).permute(2, 0, 1)
        
        input = input.to(device).unsqueeze(0)
        label = label.to(device).unsqueeze(0)
        prediction = model(input)
        if type_of_architecture == "maag" or type_of_architecture == "maag_weak":
             prediction = prediction[0]

        plot_segmentation_prediction(
                prediction.squeeze().cpu().detach().numpy().argmax(0),
                label_oh.cpu().numpy(),
                input_raw,
                label_raw,
                image_shape,
                img_name,
                join(path_to_experiment,"segmentation_prediction")
            )

        plot_segmentation_prediction_differences(
                prediction.squeeze().cpu().detach().numpy().argmax(0),
                label_oh.cpu().numpy(),
                input_raw,
                label_raw,
                image_shape,
                img_name,
                join(path_to_experiment,"segmentation_prediction_differences")
        )

        iou = classes_IoU(prediction,label)
        if type_of_architecture == "maag" or type_of_architecture == "maag_weak":
            dice = classses_dice_score(prediction, label, apply_softamax=False)
        else:
            dice = classses_dice_score(prediction, label)
        for i in range(num_class):
            ious[i].append(iou[i])
            dices[i].append(dice[i])
        mean_ious.append(np.mean(iou))
        mean_dices.append(np.mean(dices))
        accuracies.append(accuracy(prediction, label).item())
        losses.append(cross_entropy(prediction, label).item())

        cnf = confusion_matrix(
                prediction.argmax(dim=1).view(-1).cpu().numpy(),
                label_oh.argmax(dim=0).view(-1).cpu().numpy(),
                labels=np.arange(label_oh.size()[0]),
            )
        cnf_matrices.append(cnf)

    mean_dice_class = []
    mean_iou_class = []
    for i in range(num_class):
            mean_dice_class.append(np.mean(dices[i]))
            mean_iou_class.append(np.mean(ious[i]))
    results_dict = {
          "mean_iou":np.mean(mean_ious),
          "mean_dice":np.mean(mean_dices),
          "mean_loss": np.mean(losses),
          "mean_accuracy": np.mean(accuracies),
          "test_cnf_matrix_mean": np.asarray(cnf_matrices).mean(axis=0).tolist(),
          "iou_per_class": mean_iou_class,
          "dice_per_class": mean_dice_class,

    }
    with open(join(path_to_experiment, "test_results.json"), "w") as fp:
        dump(results_dict, fp)
    
