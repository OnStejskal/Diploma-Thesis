import matplotlib.pyplot as plt
import numpy as np
from numpy import array, ndarray, uint8, zeros
from os import makedirs
from os.path import join
import plotly.graph_objects as go

# Create a random 4-class segmentation tensor (you should replace this with your actual data)
# segmentation = np.random.randint(0, 4, (4, 512, 512))

# # Create a random RGB tensor (you should replace this with your actual data)
# rgb_image = np.random.rand(3, 512, 512)  # Random RGB values between 0 and 1


def plot_image(img):
    plt.imshow(np.transpose(img, (1, 2, 0)))  # Transpose the tensor to (512, 512, 3)
    plt.axis("off")
    plt.title("RGB Image")

    plt.show()


def plot_image_label(img, segmentation):
    # Define color mappings for the first plot (segmentation map)

    final_seg_mask = zeros(segmentation.shape + (3,), uint8)
    final_seg_mask[segmentation == 1] = [255, 0, 0]
    final_seg_mask[segmentation == 2] = [0, 255, 0]
    final_seg_mask[segmentation == 3] = [0, 0, 255]
    # Display the first plot (segmentation map)
    plt.subplot(1, 2, 1)
    plt.imshow(final_seg_mask)
    plt.axis("off")
    plt.title("Segmentation Map")

    # Display the second plot (RGB image)
    plt.subplot(1, 2, 2)
    plt.imshow(np.transpose(img, (1, 2, 0)))  # Transpose the tensor to (512, 512, 3)
    plt.axis("off")
    plt.title("RGB Image")

    plt.show()


def plot_segmentation(segmentation):
    # Define color mappings for the first plot (segmentation map)
    final_seg_mask = zeros(segmentation.shape + (3,), uint8)
    final_seg_mask[segmentation == 1] = [255, 0, 0]
    final_seg_mask[segmentation == 2] = [0, 255, 0]
    final_seg_mask[segmentation == 3] = [0, 0, 255]
    # Display the first plot (segmentation map)
    plt.imshow(final_seg_mask)
    plt.axis("off")
    plt.title("Segmentation Map")

    plt.show()


def create_train_val_graphs(scores: dict, experiment_path):
    makedirs(join(experiment_path, "training_validation_graphs"), exist_ok=True)
    for key, value in scores.items():
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
        else:
            continue
        fig.update_layout(xaxis_title="Epoch", yaxis_title=key)
        fig.write_image(
            join(experiment_path, "training_validation_graphs", f"{key}_history.png")
        )


def plot_mask_image_label(mask, img, segmentation):
    # Define color mappings for the first plot (segmentation map)
    plt.subplot(1, 3, 1)
    plt.imshow(np.transpose(mask, (1, 2, 0)))  # Transpose the tensor to (512, 512, 3)
    plt.axis("off")
    plt.title("masked plaque")

    final_seg_mask = zeros(segmentation.shape + (3,), uint8)
    final_seg_mask[segmentation == 1] = [255, 0, 0]
    final_seg_mask[segmentation == 2] = [0, 255, 0]
    final_seg_mask[segmentation == 3] = [0, 0, 255]
    # Display the first plot (segmentation map)
    plt.subplot(1, 3, 2)
    plt.imshow(final_seg_mask)
    plt.axis("off")
    plt.title("Segmentation Map")

    # Display the second plot (RGB image)
    plt.subplot(1, 3, 3)
    plt.imshow(np.transpose(img, (1, 2, 0)))  # Transpose the tensor to (512, 512, 3)
    plt.axis("off")
    plt.title("RGB Image")

    plt.show()
