import matplotlib.pyplot as plt
import numpy as np
from numpy import array, ndarray, uint8, zeros
import plotly.graph_objects as go
from os.path import join
from os import makedirs

import matplotlib.pyplot as plt
from numpy import array, ndarray, uint8, zeros
from PIL import Image
from skimage.segmentation import mark_boundaries

def plot_image(img):
    plt.imshow(np.transpose(img, (1, 2, 0)))  # Transpose the tensor to (512, 512, 3)
    plt.axis('off')
    plt.title('RGB Image')

    plt.show()

def plot_ae_input_output(in_segmentation, out_segmentation, img_name, save_path):
    in_segmentation_mask = zeros(in_segmentation.shape + (3,), uint8)
    in_segmentation_mask[in_segmentation == 1] = [255, 0, 0]
    in_segmentation_mask[in_segmentation == 2] = [0, 255, 0]
    in_segmentation_mask[in_segmentation == 3] = [0, 0, 255]

    out_segmentation_mask = zeros(out_segmentation.shape + (3,), uint8)
    out_segmentation_mask[out_segmentation == 1] = [255, 0, 0]
    out_segmentation_mask[out_segmentation == 2] = [0, 255, 0]
    out_segmentation_mask[out_segmentation == 3] = [0, 0, 255]

    # Display the input segmentation
    plt.subplot(1, 2, 1)
    plt.imshow(in_segmentation_mask)
    plt.axis('off')
    plt.title('Input segmentation')

    # Display the output segmentation
    plt.subplot(1, 2, 2)
    plt.imshow(out_segmentation_mask)
    plt.axis('off')
    plt.title('Output segmentation')

    plt.savefig(join(save_path, img_name))
    plt.close()


def plot_ae_input_output_segmentations(in_segmentation, out_segmentation,label, img_name, save_path):
    in_segmentation_mask = zeros(in_segmentation.shape + (3,), uint8)
    in_segmentation_mask[in_segmentation == 1] = [255, 0, 0]
    in_segmentation_mask[in_segmentation == 2] = [0, 255, 0]
    in_segmentation_mask[in_segmentation == 3] = [0, 0, 255]

    out_segmentation_mask = zeros(out_segmentation.shape + (3,), uint8)
    out_segmentation_mask[out_segmentation == 1] = [255, 0, 0]
    out_segmentation_mask[out_segmentation == 2] = [0, 255, 0]
    out_segmentation_mask[out_segmentation == 3] = [0, 0, 255]

    # Create masks for label
    label_mask = zeros(label.shape + (3,), uint8)
    label_mask[label == 1] = [255, 0, 0]
    label_mask[label == 2] = [0, 255, 0]
    label_mask[label == 3] = [0, 0, 255]

    # Plotting
    # plt.figure(figsize=(15, 5))  # Adjust the size as needed

    # Display the input segmentation
    plt.subplot(1, 3, 1)
    plt.imshow(in_segmentation_mask)
    plt.axis('off')
    plt.title('Input segmentation')

    # Display the output segmentation
    plt.subplot(1, 3, 2)
    plt.imshow(out_segmentation_mask)
    plt.axis('off')
    plt.title('Output segmentation')

    # Display the label mask
    plt.subplot(1, 3, 3)
    plt.imshow(label_mask)
    plt.axis('off')
    plt.title('Label mask')

    plt.savefig(join(save_path, img_name))
    plt.close()

def plot_segmentation_prediction_differences(
    prediction: ndarray,
    label: ndarray,
    raw_img: Image,
    raw_label: Image,
    img_shape: tuple,
    img_name: str,
    save_path: str,
) -> None:
    """

    Parameters
    ----------
    prediction : ndarray
        The segmentatnion mask predicted by a model.
    label : ndarray
        The true segmentation mask used during the training.
    raw_img : Image
        The original image.
    raw_label : Image
        The image of the original label.
    img_shape : tuple
        The shape of the network's input.
    img_name : str
        Name of the original input image file.
    save_path : str
        Path to a folder to which save the figure.
    """
    raw_img = raw_img.resize(img_shape)
    raw_img = array(raw_img)
    final_mask = mark_boundaries(raw_img, prediction == 1, [255, 0, 0])
    final_mask = mark_boundaries(final_mask, prediction == 2, [0, 255, 0])
    final_mask = mark_boundaries(final_mask, prediction == 3, [0, 0, 255])

    plaque_mask = zeros(img_shape + (3,), uint8)
    
    plaque_mask[(prediction == 2) & (label[2] == 1)] = [255, 255, 0]
    plaque_mask[(prediction == 2) & (label[2] != 1)] = [255, 0, 0]
    plaque_mask[(prediction != 2) & (label[2] == 1)] = [0, 255, 0]

    wall_mask = zeros(img_shape + (3,), uint8)
    wall_mask[(prediction == 1) & (label[1] == 1)] = [255, 255, 0]
    wall_mask[(prediction == 1) & (label[1] != 1)] = [255, 0, 0]
    wall_mask[(prediction != 1) & (label[1] == 1)] = [0, 255, 0]
    
    lumen_mask = zeros(img_shape + (3,), uint8)
    lumen_mask[(prediction == 3) & (label[3] == 1)] = [255, 255, 0]
    lumen_mask[(prediction == 3) & (label[3] != 1)] = [255, 0, 0]
    lumen_mask[(prediction != 3) & (label[3] == 1)] = [0, 255, 0]


    fig = plt.figure(figsize=(9, 3))

    fig.add_subplot(1, 3, 1)
    plt.imshow(plaque_mask)
    plt.title("Plaque")

    fig.add_subplot(1, 3, 2)
    plt.imshow(wall_mask)
    plt.title("Wall")

    fig.add_subplot(1, 3, 3)
    plt.imshow(lumen_mask)
    plt.title("Lumen")

    plt.savefig(join(save_path, img_name))
    plt.close()



def plot_segmentation_prediction(
    prediction: ndarray,
    label: ndarray,
    raw_img: Image,
    raw_label: Image,
    img_shape: tuple,
    img_name: str,
    save_path: str,
) -> None:
    """Plots the prediction and the label next to each other.

    Parameters
    ----------
    prediction : ndarray
        The segmentatnion mask predicted by a model.
    label : ndarray
        The true segmentation mask used during the training.
    raw_img : Image
        The original image.
    raw_label : Image
        The image of the original label.
    img_shape : tuple
        The shape of the network's input.
    img_name : str
        Name of the original input image file.
    save_path : str
        Path to a folder to which save the figure.
    """
    print(prediction.shape)
    print(label.shape)

    raw_img = raw_img.resize(img_shape)
    raw_img = array(raw_img)
    final_mask = mark_boundaries(raw_img, prediction == 1, [255, 0, 0])
    final_mask = mark_boundaries(final_mask, prediction == 2, [0, 255, 0])
    final_mask = mark_boundaries(final_mask, prediction == 3, [0, 0, 255])

    final_seg_mask = zeros(img_shape + (3,), uint8)
    final_seg_mask[prediction == 1] = [255, 0, 0]
    final_seg_mask[prediction == 2] = [0, 255, 0]
    final_seg_mask[prediction == 3] = [0, 0, 255]

    final_label = mark_boundaries(raw_img, label[1], [255, 0, 0])
    final_label = mark_boundaries(final_label, label[2], [0, 255, 0])

    if label.shape[0] == 4:
        final_label = mark_boundaries(final_label, label[3], [0, 0, 255])

    fig = plt.figure(figsize=(14, 14))

    fig.add_subplot(2, 2, 1)
    plt.imshow(final_mask)

    plt.title("Prediction")

    fig.add_subplot(2, 2, 2)
    plt.imshow(final_seg_mask)
    plt.title("Prediction - mask")

    fig.add_subplot(2, 2, 3)
    plt.imshow(final_label)
    plt.title("Reference")

    raw_label = array(raw_label)
    raw_label[(raw_label == [255, 255, 0]).sum(axis=2) == 3] = [255, 0, 0]
    raw_label = Image.fromarray(raw_label)
    raw_label = raw_label.resize(img_shape)
    
    fig.add_subplot(2, 2, 4)
    plt.imshow(raw_label)
    plt.title("Reference - mask")

    plt.savefig(join(save_path, img_name))
    plt.close()


def plot_image_label(img, segmentation):
        
    # Define color mappings for the first plot (segmentation map)

    final_seg_mask = zeros(segmentation.shape + (3,), uint8)
    final_seg_mask[segmentation == 1] = [255, 0, 0]
    final_seg_mask[segmentation == 2] = [0, 255, 0]
    final_seg_mask[segmentation == 3] = [0, 0, 255]
    # Display the first plot (segmentation map)
    plt.subplot(1, 2, 1)
    plt.imshow(final_seg_mask)
    plt.axis('off')
    plt.title('Segmentation Map')

    # Display the second plot (RGB image)
    plt.subplot(1, 2, 2)
    plt.imshow(np.transpose(img, (1, 2, 0)))  # Transpose the tensor to (512, 512, 3)
    plt.axis('off')
    plt.title('RGB Image')

    plt.show()

def plot_segmentation(segmentation):
        
    # Define color mappings for the first plot (segmentation map)
    
    final_seg_mask = zeros(segmentation.shape + (3,), uint8)
    final_seg_mask[segmentation == 1] = [255, 0, 0]
    final_seg_mask[segmentation == 2] = [0, 255, 0]
    final_seg_mask[segmentation == 3] = [0, 0, 255]
    # Display the first plot (segmentation map)
    
    plt.imshow(final_seg_mask)
    plt.axis('off')
    plt.title('Segmentation Map')



    plt.show()

def create_train_val_graphs(losses: dict, experiment_path):
    makedirs(join(experiment_path, "training_graphs"))

    for key, value in losses.items():
        fig = go.Figure()
        if isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
            fig.add_trace(
                go.Scatter(
                    x=[i for i in range(len(value))],
                    y=value,
                    mode="lines",
                    name=key,
                )
            )
        elif isinstance(value, list) and all(isinstance(sublist, list) and all(isinstance(x, (int, float)) for x in sublist) for sublist in value):
            for class_n in range(len(value[0])):
                class_values = [inner_list[class_n] for inner_list in value]
                fig.add_trace(
                    go.Scatter(
                        x=[i for i in range(len(class_values))],
                        y=class_values,
                        mode="lines",
                        name=f'{key}_class_{class_n}',
                    )
                )
        else:
            continue
        fig.update_layout(xaxis_title="Epoch", yaxis_title=key)
        fig.write_image(join(experiment_path, "training_graphs", f"{key}_history.png"))


    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[i for i in range(len(losses["train_losses"]))],
            y=losses["train_losses"],
            mode="lines",
            name="train_loss",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[i for i in range(len(losses["val_losses"]))],
            y=losses["val_losses"],
            mode="lines",
            name="val_loss",
        )
    )

    fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss")
    fig.write_image(join(experiment_path, "train_val_loss_history.png"))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[i for i in range(len(losses["val_ious_mean"]))],
            y=losses["val_ious_mean"],
            mode="lines",
            name="val iou",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[i for i in range(len(losses["val_dices_mean"]))],
            y=losses["val_dices_mean"],
            mode="lines",
            name="val dice coefs",
        )
    )

    fig.update_layout(xaxis_title="Epoch", yaxis_title="iou/dice score")
    fig.write_image(join(experiment_path, "iou_dice_history.png"))

