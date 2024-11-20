from generate_Synthetic_data import (
    generate_synthetic_image_transversal,
    generate_synthetic_image_longitudal,
    create_random_image_from_segmentation,
)
import numpy as np
from datetime import datetime
import os
from PIL import Image
import json
import warnings

warnings.filterwarnings("ignore")


def create_N_images(N, folder_path=None, mode="transversal", plaque_mean = 0.5):
    """Function to create synthetic images

    Args:
        N (int): number of images
        folder_path (str, optional): path to the dataset. Defaults to None.
        mode (str, optional): orientation of images. Defaults to 'transversal'.
    """

    now = datetime.now()
    current_time = now.strftime("%m-%d_%H-%M")
    if folder_path is None:
        folder_path = f"images_from_{current_time}_number_{N}"
    folder_path = os.path.join("synthetic_pictures", folder_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path,  exist_ok=True)
        print("Folder created successfully!")
    else:
        print("Folder already exists.")

    segmentation_images_path = os.path.join(folder_path, "segmented_images")
    labels_images_path = os.path.join(folder_path, "labels")
    raw_labels_images_path = os.path.join(labels_images_path, "raw_labels")
    random_images_path = os.path.join(folder_path, "random_images")
    random_images_path_png = os.path.join(folder_path, "random_images_png")
    segmentation_images_path_png = os.path.join(folder_path, "segmented_images_png")
    os.makedirs(segmentation_images_path, exist_ok=True)
    os.makedirs(labels_images_path, exist_ok=True)
    os.makedirs(raw_labels_images_path,  exist_ok=True)
    os.makedirs(random_images_path, exist_ok=True)
    os.makedirs(random_images_path_png, exist_ok=True)
    os.makedirs(segmentation_images_path_png, exist_ok=True)

    for i in range(N):
        if i % 10 == 0:
            print(i)
        seg_img, parameters = (
            generate_synthetic_image_transversal(0)
            if mode == "transversal"
            else generate_synthetic_image_longitudal(3, 3, None, False, True, False)
        )
        random_image = create_random_image_from_segmentation(seg_img, plaque_mean = plaque_mean)

        pil_seg_img = Image.fromarray(seg_img.astype("uint8"), "RGB")
        pil_seg_img.save(os.path.join(segmentation_images_path_png, f"image_{i}.png"))
        pil_rand_img = Image.fromarray(random_image.astype("uint8"), "L")
        pil_rand_img.save(os.path.join(random_images_path_png, f"image_{i}.png"))

        np.save(os.path.join(segmentation_images_path, f"image_{i}.npy"), seg_img)
        np.save(os.path.join(random_images_path, f"image_{i}.npy"), random_image)
        with open(
            os.path.join(raw_labels_images_path, f"image_{i}.json"), "w"
        ) as json_file:
            json.dump(parameters, json_file)


create_N_images(250,folder_path="echogenecity0", mode="transversal", plaque_mean = 0.9)
create_N_images(250,folder_path="echogenecity1", mode="transversal", plaque_mean = 0.7)
create_N_images(250,folder_path="echogenecity2", mode="transversal", plaque_mean = 0.5)
create_N_images(250,folder_path="echogenecity3", mode="transversal", plaque_mean = 0.3)
