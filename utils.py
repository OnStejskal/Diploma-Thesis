import os
import shutil
from os.path import join
from PIL import Image

def create_folder_structure_for_parameters_computation(images_folder, labels_folder, dir_path, train_ratio=0.8, val_ratio = 0.1):
    # Get list of filenames in both folders
    image_files = os.listdir(images_folder)
    label_files = os.listdir(labels_folder)

    # Sort the filenames to ensure correspondence
    image_files.sort()
    label_files.sort()

    # Calculate the number of files for training
    num_train = int(train_ratio * len(image_files))
    num_val = int((train_ratio+val_ratio) * len(image_files))

    # Create train and validation folders if not exists
    os.makedirs(dir_path)

    os.makedirs(join(dir_path, "train"))
    os.makedirs(join(dir_path, "val"))

    train_images_path =join(dir_path, "train", "images")
    val_images_path = join(dir_path, "val","images")
    train_segmentaion_path = join(dir_path, "train", "segmentations")
    val_segmentations_path = join(dir_path, "val", "segmentations")

    test_images_path =join(dir_path, "test", "images")
    test_segmentations_path =join(dir_path, "test", "segmentations")


    os.makedirs(train_images_path)
    os.makedirs(val_images_path)
    os.makedirs(train_segmentaion_path)
    os.makedirs(val_segmentations_path)
    os.makedirs(test_images_path)
    os.makedirs(test_segmentations_path)

    # Copy images to train folder
    for image_file in image_files[:num_train]:
        src_image = os.path.join(images_folder, image_file)
        dst_image = os.path.join(train_images_path, image_file)
        shutil.copy(src_image, dst_image)

    # Copy labels to train folder
    for label_file in label_files[:num_train]:
        src_label = os.path.join(labels_folder, label_file)
        dst_label = os.path.join(train_segmentaion_path, label_file)
        shutil.copy(src_label, dst_label)

    # Copy remaining images to validation folder
    for image_file in image_files[num_train:num_val]:
        src_image = os.path.join(images_folder, image_file)
        dst_image = os.path.join(val_images_path, image_file)
        shutil.copy(src_image, dst_image)

    # Copy remaining labels to validation folder
    for label_file in label_files[num_train:num_val]:
        src_label = os.path.join(labels_folder, label_file)
        dst_label = os.path.join(val_segmentations_path, label_file)
        shutil.copy(src_label, dst_label)


    # Copy remaining images to validation folder
    for image_file in image_files[num_val:]:
        src_image = os.path.join(images_folder, image_file)
        dst_image = os.path.join(test_images_path, image_file)
        shutil.copy(src_image, dst_image)

    # Copy remaining labels to validation folder
    for label_file in label_files[num_val:]:
        src_label = os.path.join(labels_folder, label_file)
        dst_label = os.path.join(test_segmentations_path, label_file)
        shutil.copy(src_label, dst_label)



def create_folder_structure_train_val_test_for_synthetic_data():
    
    # Base directory where the folders are located
    base_dir = "image_generatio/synthetic_pictures"

    # Sub-directories for different echogenicities
    sub_dirs = ["echogenecity0", "echogenecity1", "echogenecity2", "echogenecity3"]

    # Destination directory
    dest_dir = "parameters_computation/data/synthetic/images"

    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    # Process each sub-directory
    for sub_dir in sub_dirs:
        # Determine the echogenicity number
        echogenicity_number = sub_dir[-1]

        # Full path to the sub-directory
        full_path = os.path.join(base_dir, sub_dir, "random_images_png")

        # List all files in the directory
        for filename in os.listdir(full_path):
            # Check if it's an image
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                # Generate a new filename with the echogenicity number
                new_filename = f"{echogenicity_number}_{filename}"

                # Path of the source file
                source_file = os.path.join(full_path, filename)

                # Path of the destination file
                dest_file = os.path.join(dest_dir, new_filename)

                # Copy the file to the destination directory
                shutil.copy2(source_file, dest_file)

    print("Images have been copied and renamed.")


    # Base directory where the folders are located
    base_dir = "image_generatio/synthetic_pictures"

    # Sub-directories for different echogenicities
    sub_dirs = ["echogenecity0", "echogenecity1", "echogenecity2", "echogenecity3"]

    # Destination directory
    dest_dir = "parameters_computation/data/synthetic/segmentations"

    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    # Process each sub-directory
    for sub_dir in sub_dirs:
        # Determine the echogenicity number
        echogenicity_number = sub_dir[-1]

        # Full path to the sub-directory
        full_path = os.path.join(base_dir, sub_dir, "segmented_images_png")

        # List all files in the directory
        for filename in os.listdir(full_path):
            # Check if it's an image
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                # Generate a new filename with the echogenicity number
                new_filename = f"{echogenicity_number}_{filename}"

                # Path of the source file
                source_file = os.path.join(full_path, filename)

                # Path of the destination file
                dest_file = os.path.join(dest_dir, new_filename)

                # Copy the file to the destination directory
                shutil.copy2(source_file, dest_file)

    print("Images have been copied and renamed.")
                
    import os
    import shutil
    import random

    # Path to the folder with all images
    all_images_dir = "parameters_computation/data/synthetic/images"
    all_seg_dir = "parameters_computation/data/synthetic/segmentations"

    # Paths for train, validation, and test folders
    train_dir = "parameters_computation/data/synthetic/train/images"
    val_dir = "parameters_computation/data/synthetic/val/images"
    test_dir = "parameters_computation/data/synthetic/test/images"
    strain_dir = "parameters_computation/data/synthetic/train/segmentations"
    sval_dir = "parameters_computation/data/synthetic/val/segmentations"
    stest_dir = "parameters_computation/data/synthetic/test/segmentations"

    # Create the directories if they don't exist
    for directory in [train_dir, val_dir, test_dir, strain_dir, sval_dir,stest_dir ]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # List all image files
    all_images = [f for f in os.listdir(all_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]

    # Shuffle the list for random distribution
    random.shuffle(all_images)

    # Calculate split indices
    total_images = len(all_images)
    train_split_index = int(0.8 * total_images)
    val_split_index = train_split_index + int(0.1 * total_images)

    # Split the images
    train_images = all_images[:train_split_index]
    val_images = all_images[train_split_index:val_split_index]
    test_images = all_images[val_split_index:]

    # Function to copy files to respective folders
    def copy_files1(images, source,destination):
        for image in images:
            shutil.copy2(os.path.join(source, image), os.path.join(destination, image))
    def copy_files2(images, source,destination):
        for image in images:
            im = images
            shutil.copy2(os.path.join(source, image), os.path.join(destination, image))



    # Copy images to respective folders
    copy_files1(train_images,all_images_dir, train_dir)
    copy_files1(val_images,all_images_dir, val_dir)
    copy_files1(test_images,all_images_dir, test_dir)

    copy_files2(train_images,all_seg_dir, strain_dir)
    copy_files2(val_images,all_seg_dir, sval_dir)
    copy_files2(test_images,all_seg_dir, stest_dir)

    print("Images have been distributed into train, validation, and test folders.")

def create_squared_crop_coordinates_img(coords, img, pad_size = 5):
        x1, y1, x2, y2 = coords.astype(int)   
        width = x2 - x1
        height = y2 - y1
        if width >= height:
            diff = width - height
            y1 = y1 - diff//2
            y2 = y2 + diff // 2 + (diff%2)
        else:
            diff = height - width
            x1 = x1 - diff//2
            x2 = x2 + diff // 2 + (diff%2)
        if x1 - pad_size >= 0 and y1 - pad_size >= 0 and x2 + pad_size < img.size[0] and y2 + pad_size < img.size[1]:
            x1 = x1 - pad_size
            y1 = y1 - pad_size
            x2 = x2 + pad_size
            y2 = y2 + pad_size
        else:
            print("WARNING NOT PADDING")

        new_size = x2 - x1
        final_image = Image.new('RGB', (new_size, new_size), (0, 0, 0))
        cropped_image = img.crop((x1, y1, x2, y2))
        paste_x = (new_size - (x2 - x1)) // 2
        paste_y = (new_size - (y2 - y1)) // 2
        final_image.paste(cropped_image, (paste_x, paste_y))
        return final_image

def create_fix_crop_coordinates_img(coords, img):
        orig_w, orig_h = img.size
        print("imgsize ", img.size)
        x1, y1, x2, y2 = coords.astype(int)
        new_size = 380   

        width = x2 - x1
        height = y2 - y1

        if width >= new_size:
            diff = width - new_size
            x1 = x1 + diff//2
            x2 = x2 - diff // 2 + (diff%2)
        else:
            diff = new_size - width
            x1 = x1 - diff//2
            x2 = x2 + diff // 2 + (diff%2)


        if height >= new_size:
            diff = width - new_size
            y1 = y1 + diff//2
            y2 = y2 - diff // 2 + (diff%2)
        else:
            diff = new_size - width
            y1 = y1 - diff//2
            y2 = y2 + diff // 2 + (diff%2)

        x1 = max(0, x1)
        x2 = min(x2, orig_w)

        y1 = max(0, y1)
        y2 = min(y2, orig_h)



        if(x1 >= x2 or y1 >= y2):
            print("ERROR negative width ot height")
            return None

        # if x1 - pad_size >= 0 and y1 - pad_size >= 0 and x2 + pad_size < img.size[0] and y2 + pad_size < img.size[1]:
        #     x1 = x1 - pad_size
        #     y1 = y1 - pad_size
        #     x2 = x2 + pad_size
        #     y2 = y2 + pad_size
        # else:
        #     print("WARNING NOT PADDING")

        final_image = Image.new('RGB', (new_size, new_size), (0, 0, 0))
        cropped_image = img.crop((x1, y1, x2, y2))
        paste_x = (new_size - (x2 - x1)) // 2
        paste_y = (new_size - (y2 - y1)) // 2
        final_image.paste(cropped_image, (paste_x, paste_y))
        return final_image
