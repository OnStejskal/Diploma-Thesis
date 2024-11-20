import cv2
import numpy as np
from skimage import measure


def compute_proportion_green_blue(image):
    """Function to compute ratio between plaque area and vein area

    Args:
        image (ndarray): 3 dimnensional RGB image of segmentation as numpy array
    Returns:
        green_blue_proportion (float): proporiotn of green pixels to sum of green and blue
    """
    plaque_mask = np.all(image == [0, 255, 0], axis=-1)
    lumen_mask = np.all(image == [0, 0, 255], axis=-1)
    N_green = np.count_nonzero(plaque_mask)
    N_blue = np.count_nonzero(lumen_mask)
    return float(N_green / (N_blue + N_green))


def measure_area_of_segment(image, segment_name="plaque"):
    """Function to measure area of the segment in the pixels

    Args:
        image (ndarray): 3 dimnensional RGB image of segmentation as numpy array
        segment_name (str, optional): which segments area is measured . Defaults to "plaque".
            - options:
                - wall
                - lumen
                - plaque

    Returns:
        float: area in pixels
    """

    if segment_name == "wall":
        dot_product_mask = [1.0, 0.0, 0.0]
    if segment_name == "lumen":
        dot_product_mask = [0.0, 0.0, 1.0]
    if segment_name == "plaque":
        dot_product_mask = [0.0, 1.0, 0.0]

    gray_image = np.dot(image[..., :3], dot_product_mask)
    # Count the number of non-zero pixels in the grayscale image
    area = np.count_nonzero(gray_image)
    return area


def most_distant_points_in_segment(image, segment_name="wall"):
    """function that computes the diameter (two most distant points in the segment) of the segment

    Args:
        image (ndarray): 3 dimnensional RGB image of segmentation as numpy array
        segment_name (str, optional): which segment to measure [wall, lumen, plaque]. Defaults to "wall".

    Returns:
        distance (float): distance between two most distant points of the segment
    """
    mask_vector = [255, 0, 0]
    if segment_name == "lumen":
        mask_vector = [0, 0, 255]
    elif segment_name == "plaque":
        mask_vector = [0, 255, 0]

    mask = np.all(image == mask_vector, axis=-1)
    contours = measure.find_contours(mask, 0.5)
    # Get the longest contour
    if not contours:
        return 0
    longest_contour = max(contours, key=len)
    longest_distance = 0
    for i in range(len(longest_contour)):
        for j in range(i + 1, len(longest_contour)):
            distance = np.sqrt(
                (longest_contour[i][0] - longest_contour[j][0]) ** 2
                + (longest_contour[i][1] - longest_contour[j][1]) ** 2
            )
            if distance > longest_distance:
                longest_distance = distance
    return distance


def circumscribed_circle_around_biggest_plaque(image, segment_name="wall"):
    """measure the diameter of the circle circumscribing the segment (kruznice opsana) measure the circle around biggest plaque

    Args:
        image (ndarray): 3 dimnensional RGB image of segmentation as numpy array
        segment_name (str, optional): which segment to measure [wall, lumen, plaque]. Defaults to "wall".

    Returns:
        diameter (float): diameter of the circumscribed circle
    """

    channel = image[:, :, 0]
    if segment_name == "lumen":
        channel = image[:, :, 2]
    elif segment_name == "plaque":
        channel = image[:, :, 1]

    # Threshold the green channel to obtain a binary mask of the green pixels
    _, mask = cv2.threshold(channel, 10, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the contour with the largest area
    if not contours:
        return 0
    contour = max(contours, key=cv2.contourArea)

    # Compute the minimum enclosing circle of the selected contour
    _, radius = cv2.minEnclosingCircle(contour)

    diameter = radius * 2
    return diameter


def compute_longitudal_params(image):
    plaque_mask = np.all(image == [0, 255, 0], axis=-1)
    plaque_mask_t = np.transpose(plaque_mask)
    lumen_mask = np.all(image == [0, 0, 255], axis=-1)
    lumen_mask_t = np.transpose(lumen_mask)

    min_lumen_width = np.inf

    max_plaque_width = -np.inf

    max_ratio = -np.inf

    lumen_width_sum = 0
    plaque_width_sum = 0
    ratio_sum = 0
    nonzero_cols = 0

    for i, (col_plaque, col_lumen) in enumerate(zip(plaque_mask_t, lumen_mask_t)):
        # find the first and last non-zero indices in the row
        plaque_column = np.nonzero(col_plaque)[0]
        if plaque_column.size > 0:
            lumen_column = np.nonzero(col_lumen)[0]

            nonzero_cols += 1
            lumen_width = len(lumen_column)
            plaque_width = len(plaque_column)
            ratio = plaque_width / min_lumen_width
            lumen_width_sum += lumen_width
            plaque_width_sum += plaque_width
            ratio_sum += ratio

            if lumen_width < min_lumen_width:
                min_lumen_width = lumen_width
            if plaque_width > max_plaque_width:
                max_plaque_width = plaque_width
            if ratio > max_ratio:
                max_ratio = ratio

    return {
        "min_lumen_width": min_lumen_width,
        "max_plaque_width": max_plaque_width,
        "max_ratio": max_ratio,
        "avg_lumen_width": lumen_width_sum / nonzero_cols,
        "avg_plaque_width": plaque_width_sum / nonzero_cols,
        "avg_ratio": ratio_sum / nonzero_cols,
        "plaque_length": nonzero_cols,
        "proportion_green_blue": compute_proportion_green_blue(image),
        # "plaque_circle_diameter": circumscribed_circle(image, segment_name="plaque"),
        "wall_area": measure_area_of_segment(image, segment_name="wall"),
        "lumen_area": measure_area_of_segment(image, segment_name="lumen"),
        "plaque_area": measure_area_of_segment(image, segment_name="plaque"),
    }


def biggest_plaque_params(image):
    """_summary_

    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    lower_green = np.array([0, 120, 0])
    upper_green = np.array([80, 255, 80])
    mask = cv2.inRange(image, lower_green, upper_green)
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the contour with the largest area
    if len(contours) == 0:
        return {
            "max_plaque_area": 0,
        }
    largest_contour = max(contours, key=cv2.contourArea)
    # Create a mask for the largest contour
    largest_contour_mask = np.zeros_like(mask)
    cv2.drawContours(
        largest_contour_mask, [largest_contour], 0, 255, thickness=cv2.FILLED
    )
    # Apply the mask to the original image to extract the largest green object
    largest_green_object = cv2.bitwise_and(image, image, mask=largest_contour_mask)

    # cv2.imshow("Largest Green Object", largest_green_object)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return {"max_plaque_area": measure_area_of_segment(np.array(largest_green_object))}


def compute_plaque_lumen_passage_params(image):
    """Function to compute parameters regarding reduced passage in longitudual images

    This function creates parameters of the part of longitudional image where is plaque based.
    It computes the maximal average plaque heighht, minimal and average lumen height and maximal and average ratios,
    between plaque and lumen.Also the width of the plaque

    Args:
        image (ndarray): 3 dimnensional RGB image of segmentation as numpy array

    Returns:
        parameters (dictionary): parameters about space between plaque and lumen
    """
    plaque_lower_bound = np.array([0, 120, 0])
    plaque_upper_bound = np.array([80, 255, 80])

    # Create a mask based on the color range
    plaque_mask = np.logical_and(
        np.all(image >= plaque_lower_bound, axis=-1),
        np.all(image <= plaque_upper_bound, axis=-1),
    )
    plaque_mask_t = np.transpose(plaque_mask)

    lumen_lower_bound = np.array([0, 0, 120])
    lumen_upper_bound = np.array([80, 80, 255])

    # Create a mask based on the color range
    lumen_mask = np.logical_and(
        np.all(image >= lumen_lower_bound, axis=-1),
        np.all(image <= lumen_upper_bound, axis=-1),
    )
    lumen_mask_t = np.transpose(lumen_mask)

    min_lumen_width = 256
    max_plaque_width = 0
    max_ratio = 0
    lumen_width_sum = 0
    plaque_width_sum = 0
    ratio_sum = 0
    nonzero_cols = 0

    for col_plaque, col_lumen in zip(plaque_mask_t, lumen_mask_t):
        # find the first and last non-zero indices in the row
        plaque_column = np.nonzero(col_plaque)[0]

        # computation done only for columns including plaque
        if plaque_column.size > 0:
            lumen_column = np.nonzero(col_lumen)[0]

            # ommit columns where is only plaque and no lumen to be pushed by plaque
            if lumen_column.size == 0:
                continue
            nonzero_cols += 1
            lumen_width = len(lumen_column)
            plaque_width = len(plaque_column)
            ratio = plaque_width / min_lumen_width
            lumen_width_sum += lumen_width
            plaque_width_sum += plaque_width
            ratio_sum += ratio

            if lumen_width < min_lumen_width:
                min_lumen_width = lumen_width
            if plaque_width > max_plaque_width:
                max_plaque_width = plaque_width
            if ratio > max_ratio:
                max_ratio = ratio

    return {
        "min_lumen_width": min_lumen_width,
        "max_plaque_width": max_plaque_width,
        "max_ratio": max_ratio,
        "avg_lumen_width": lumen_width_sum / nonzero_cols if nonzero_cols != 0 else 0,
        "avg_plaque_width": plaque_width_sum / nonzero_cols if nonzero_cols != 0 else 0,
        "avg_ratio": ratio_sum / nonzero_cols if nonzero_cols != 0 else 0,
        "plaque_length": nonzero_cols,
    }


def compute_longitudal_geometrical_params(image):
    """Function to create all geometrical longitudinal parameters from segmentations

    Args:
        image (ndarray): 3 dimnensional RGB image of segmentation as numpy array

    Returns:
        longitudinal_parameters (dictionary): dictionary of geometrical longitudinal parameters
    """
    parameters = compute_plaque_lumen_passage_params(image)
    other_parameters = {
        "proportion_green_blue": compute_proportion_green_blue(image),
        "plaque_circle_diameter": circumscribed_circle_around_biggest_plaque(
            image, segment_name="plaque"
        ),
        "wall_area": measure_area_of_segment(image, segment_name="wall"),
        "lumen_area": measure_area_of_segment(image, segment_name="lumen"),
        "plaque_area": measure_area_of_segment(image, segment_name="plaque"),
    }
    parameters.update(other_parameters)
    return parameters


def compute_transversal_geometrical_params(image):
    """Function to create all geometrical transversal parameters from segmentations

     Args:
        image (ndarray): 3 dimnensional RGB image of segmentation as numpy array

    Returns:
        transversal_parameters (dictionary): dictionary of geometrical longitudinal parameters
    """

    parameters = {
        "proportion_green_blue": compute_proportion_green_blue(image),
        "wall_area": measure_area_of_segment(image, segment_name="wall"),
        "lumen_area": measure_area_of_segment(image, segment_name="lumen"),
        "plaque_area": measure_area_of_segment(image, segment_name="plaque"),
        "wall_circle": circumscribed_circle_around_biggest_plaque(
            image, segment_name="wall"
        ),
        "plaque_circle": circumscribed_circle_around_biggest_plaque(
            image, segment_name="plaque"
        ),
        "lumen_circle": circumscribed_circle_around_biggest_plaque(
            image, segment_name="lumen"
        ),
    }

    parameters.update(biggest_plaque_params(image))

    return parameters
