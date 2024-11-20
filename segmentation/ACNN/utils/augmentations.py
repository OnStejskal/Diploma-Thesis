import torch

def swap_pixels_batch(segmentation_batch: torch.tensor, swap_prob=0.1):
    """label noise layer where randomly with probability of 0.1 label is swapped to another class

    Args:
        segmentation_batch (torch.tensor): batch of segmentations on which the operation is performed
        swap_prob (float, optional): pobability on which to swap labels. Defaults to 0.1.

    Returns:
        torch.tensor: batch of segmentations with randomly swapped pixels
    """
    # Process each segmentation in the batch
    swapped_batch = []
    for segmentation in segmentation_batch:
        # Flatten the segmentation tensor
        flattened_segmentation = segmentation.flatten()

        # Calculate the number of pixels to swap
        num_pixels_to_swap = int(flattened_segmentation.numel() * swap_prob)

        # Randomly select pixel indices
        indices_to_swap = torch.randperm(flattened_segmentation.numel())[:num_pixels_to_swap]

        # Get unique classes
        unique_classes = torch.unique(segmentation)

        # Assign a random class to each selected pixel
        for idx in indices_to_swap:
            current_class = flattened_segmentation[idx]
            new_classes = unique_classes[unique_classes != current_class]
            flattened_segmentation[idx] = new_classes[torch.randint(len(new_classes), (1,))]

        # Reshape back to the original shape
        swapped_segmentation = flattened_segmentation.reshape(segmentation.shape)
        swapped_batch.append(swapped_segmentation)

    # Stack all swapped segmentations to form a batch
    swapped_batch_tensor = torch.stack(swapped_batch)

    return swapped_batch_tensor


def oh_swap_pixels_in_batch(tensor, swap_prob=0.1):
    """label noise layer where randomly with probability of 0.1 label is swapped to another class

    Args:
        segmentation_batch (torch.tensor): batch of one hot encoded segmentations on which the operation is performed
        swap_prob (float, optional): pobability on which to swap labels. Defaults to 0.1.

    Returns:
        torch.tensor: batch of one hot encoded segmentations with randomly swapped pixels
    """
    # Assuming tensor shape is (batch, num_classes, height, width)
    batch_size, num_classes, height, width = tensor.shape

    # Process each item in the batch separately
    swapped_batch = []
    for i in range(batch_size):
        # Flatten the spatial dimensions
        flattened = tensor[i].view(num_classes, -1)  # Shape: [num_classes, height*width]

        # Calculate the number of pixels to swap
        num_pixels_to_swap = int(height * width * swap_prob)

        # Randomly select pixel indices
        indices_to_swap = torch.randperm(height * width)[:num_pixels_to_swap]

        # Swap pixels
        for idx in indices_to_swap:
            # Randomly choose a new class for this pixel, different from the current class
            current_class = torch.argmax(flattened[:, idx])
            possible_classes = [c for c in range(num_classes) if c != current_class]
            new_class = possible_classes[torch.randint(0, len(possible_classes), (1,)).item()]
            new_pixel = torch.eye(num_classes)[:, new_class]
            
            # Update the pixel in flattened tensor
            flattened[:, idx] = new_pixel

        # Reshape back to original shape and add to swapped batch
        swapped_item = flattened.view(num_classes, height, width)
        swapped_batch.append(swapped_item)

    # Reassemble the batch
    swapped_tensor = torch.stack(swapped_batch, dim=0)

    return swapped_tensor