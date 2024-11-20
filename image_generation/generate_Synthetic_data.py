import cv2
import random
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from datAugSingleInput import *
from scipy.ndimage import gaussian_filter
from compute_metrics import compute_longitudal_params, compute_transversal_geometrical_params

def create_random_image_from_segmentation(seg_image, sigma = 3, show = False, plaque_mean = 0.5):
    """This function creates random image from the segmented image

    Creation of grayscale image where the intensities are sampled from the normal distribution, 
    which parameters depends on sampled segment. The image is also filtered by gaussian filter 
    
    Args:
        seg_image (ndarray): 3 dimnensional RGB image of segmentation as numpy array
        sigma (int, optional): sigma for the gauss filter. Defaults to 3.
        show (bool, optional): pyplot show the image. Defaults to False.

    Returns:
        numpyArray: 2d numpy array representing the grayscale random image
    """
    # Define mean and standard deviation of Gaussian distribution
    BLUE_MEAN = 0
    BLUE_STD = 0.1

    RED_MEAN = 0.3
    RED_STD = 0.1

    GREEN_MEAN = plaque_mean
    GREEN_STD = 0.15

    BLACK_MEAN = 0.4
    BLACK_STD = 0.3

    new_image = np.zeros(seg_image.shape[:2])

    rgb_values = [[255, 0, 0],[0, 255, 0],[0, 0, 255], [0, 0, 0]]
    means = [RED_MEAN, GREEN_MEAN, BLUE_MEAN, BLACK_MEAN]
    stds = [RED_STD, GREEN_STD, BLUE_STD, BLACK_STD]

    # Assigning random values
    for rgb_value, mean, std in zip(rgb_values, means, stds):
        mask = np.all(seg_image == rgb_value, axis=2)
        random_values = np.abs(np.random.normal(mean, std, size=np.sum(mask)))
        new_image[mask] = random_values

    # Rescale values to range [0, 1]
    # new_image = (new_image - np.min(new_image)) / (np.max(new_image) - np.min(new_image))

    # Convert to uint8
    new_image = (new_image * 255).astype(np.uint8)
    
    # Apply the Gaussian filter to the image
    new_image = gaussian_filter(new_image, sigma=sigma)

    if show:
        plt.imshow(new_image, cmap='gray')
        plt.show()

    return new_image



def in_ellipse(x, y ,center_x, center_y, a, b):
    """Function helps to test if is the point (x, y) inside ellipse with center (center_x, center_y ) and (a, b) are major and minor radius

    Args:
        x (int): x coord
        y (int): y coord
        center_x (int): center x coord
        center_y (int): center y coord
        a (float): major ellipse radius
        b (float): minor ellipse radius

    Returns:
        bool: signals whetrher point x, y is inside ellpise
    """
    return (y - center_y)**2 / b**2 + (x - center_x)**2 / a**2 < 1


def rotate(image, angle):
    """Functtion to rotate iumage

    Args:
        image (cv2.Image): image to be rotated
        angle (float): angle to rotate

    Returns:
        cv2.Image: rotated image
    """

    RotMatrix = cv2.getRotationMatrix2D((image.shape[0]//2, image.shape[1]//2), angle = angle, scale=1.0)
    rotated_image = cv2.warpAffine(image, RotMatrix, (image.shape[0], image.shape[1]))
    return rotated_image

def generate_synthetic_image_transversal(transformation_mode = 0, show = False):
    """ Function creates an Tranversal image.
    
    Creation of transversal image for synthetic dataset by using ellipses and geometric transformations. 
    This function is mainly code from Artem Moroz project: https://gitlab.fel.cvut.cz/morozart/bachelor-thesis/-/blob/main/generate_Synthetic_data.py

    Args:
        transformation_mode (int, optional): transformation_mode of . Defaults to 0. 
            - 0 all transformations
            - 1 grid distortion
            - 2 elastic transformation
            - 3 no transformation
        show (boolean, optional): indicates whether to plot the image. Defaults to False. 

    Returns:
        numpy array (W x H x 3): RGB image as numpy array
    """
    base_image = np.zeros((256, 256, 3))
    center_artery_y = randint(115, 140)
    center_artery_x = randint(115, 140)
    artery_diam_y = randint(90, 110)
    artery_diam_x = artery_diam_y + randint(-10, 10)
    plaque_diam_y = artery_diam_y - 10
    plaque_diam_x = artery_diam_x - 10
    lumen_diam_x = randint(int(plaque_diam_x * 0.3), int(plaque_diam_x * 0.8))
    while True:
        lumen_diam_y = randint(int(lumen_diam_x * 0.3), int(0.8 * lumen_diam_x))
        if lumen_diam_y != 0:
            break

    if lumen_diam_x > lumen_diam_y:
        lumen_center_x = center_artery_x
        lumen_center_y = center_artery_y + plaque_diam_y - lumen_diam_y
    else:
        lumen_center_y = center_artery_y
        lumen_center_x = center_artery_x + plaque_diam_x - lumen_diam_x
    is_green = random.random()
    for x in range(256):
        for y in range(256):
            if True:
                if in_ellipse(x, y, center_artery_x, center_artery_y, artery_diam_x, artery_diam_y) and not in_ellipse(x, y, center_artery_x, center_artery_y, plaque_diam_x, plaque_diam_y) \
                        and not in_ellipse(x, y, lumen_center_x, lumen_center_y, lumen_diam_x, lumen_diam_y):
                    base_image[x, y, :] = [255, 0, 0] # fill walls with red color
                elif in_ellipse(x, y, center_artery_x, center_artery_y, plaque_diam_x, plaque_diam_y) and not in_ellipse(x, y, lumen_center_x, lumen_center_y, lumen_diam_x, lumen_diam_y):
                    base_image[x, y, :] = [0, 0, 255] # fill lumen with green color
                elif in_ellipse(x, y, lumen_center_x, lumen_center_y, lumen_diam_x, lumen_diam_y):
                    base_image[x, y, :] = [0, 255, 0] # fill plaque location with green color
            else: # 15 % do not have any plaque.
                if in_ellipse(x, y, center_artery_x, center_artery_y, artery_diam_x, artery_diam_y) and not in_ellipse(x, y, center_artery_x, center_artery_y, plaque_diam_x, plaque_diam_y) \
                        and not in_ellipse(x, y, lumen_center_x, lumen_center_y, lumen_diam_x, lumen_diam_y):
                    base_image[x, y, :] = [255, 0, 0]
                elif in_ellipse(x, y, center_artery_x, center_artery_y, plaque_diam_x, plaque_diam_y):
                    base_image[x, y, :] = [0, 0, 255]

    apply_rot = random.random()
    if apply_rot > 0.5:
        angle = 360 * random.random()
        base_image=rotate(base_image, angle)

    if transformation_mode == 0 or transformation_mode == 1:
        base_image = Grid_distortion(base_image) 
    if transformation_mode == 0 or transformation_mode ==2:
        base_image = Elastic_transform(base_image)

    # after grid distortion and elastic transform application some pixels lying close to margins between different colors change their color to some intermediate color. So I can restore the initial
    # color by knowing value of the dot product between actual pixel intensity and RGB color code.
    for x in range(256):
        for y in range(256):
            if not np.array_equal(base_image[x, y], [0., 255., 0.]) and not np.array_equal(base_image[x, y], [255., 0., 0.]) \
                and not np.array_equal(base_image[x, y], [0., 0., 255.]) and not np.array_equal(base_image[x, y], [0., 0., 0.]):
                if np.dot(base_image[x, y], [0., 255., 0.])/(np.linalg.norm(base_image[x, y]) *np.linalg.norm([0., 255., 0.]) ) >= 0.65:
                    base_image[x, y, :] = [0, 255, 0]
                elif np.dot(base_image[x, y], [0., 0., 255.])/(np.linalg.norm(base_image[x, y]) *np.linalg.norm([0., 0., 255.]) ) >= 0.65:
                    base_image[x, y, :] = [0, 0, 255]
                else:
                    base_image[x, y, :] = [255, 0, 0]
    if show:
        plt.imshow(base_image)
        plt.show()

    # Compute geometrical parameters from the image
    parameters_measurements = compute_transversal_geometrical_params(base_image)

    return base_image.astype(np.uint8), parameters_measurements


def generate_random_heights_of_plaque(size,height):
    """Util function to create plaque heights by a random increasing and decreasing series

    Args:
        size (int): number of points in horizontal direction
        height (int): approximated maximal height of the plaque

    Returns:
        list(int): list of the heights for each point 
    """
    total_increasing = size//2
    total_decreasing = size - total_increasing

    increasing_step = round(height / total_increasing)

    values = []
    current_value = 0

    for _ in range(total_increasing):
        current_value += randint(increasing_step//2, increasing_step*2)
        values.append(round(current_value))

    for _ in range(total_decreasing):
        current_value -= randint(increasing_step//2, increasing_step*2)
        current_value = max(1, current_value)
        values.append(current_value)
    return values


def generate_synthetic_image_longitudal(num_points, points_multiplier_for_interpolation = 3, polynom_degree = None, deform = False, meassure = True, show = False):
    """Creates synthetic longitudal image using two polynoms as lines

    Creates random longitudinal picture by using the polynoms to create edges of vein and creates random plaque by using random series of heights

    Args:
        num_points (int): number of flex points on each side of the vein 
        points_multiplier_for_interpolation (int, optional): multiplication factor of new points that are used for the interpolation. Defaults to 3.
        polynom_degree (int, optional): interpolation polynom degree. Defaults to None in that case polynom_degree = num_points + 2.
        deform (bool, optional): indicates whether to perform deformation. Defaults to False.
        meassure (bool, optional): indicates whether to meassure the parameters. Defaults to True.
        show (bool, optional): indicates whether to plot the image. Defaults to False.

    Returns:
        tuple(segmented_image, meassurements)
        numpy_array: segmented_image
        dictionary: meassurements

    """

    major_shift = 30
    minor_shift = 10

    base_image = np.zeros((256, 256, 3))

    # creates basic random parameters of the generated image 
    l_width = randint(120, 150)
    r_width = l_width + randint(-20,20) 
    l_center = 256/2 + randint(-20, 20)
    r_center = l_center + randint(-major_shift, major_shift)

    # creates the four points that defines the rectangle of the vein
    l_u_y = int(max(0, l_center - l_width/2))
    l_d_y = int(min(255, l_center + l_width/2))
    r_u_y = int(max(0, r_center - r_width/2))
    r_d_y = int(min(255, r_center + r_width/2))

    samples = np.linspace(0, 256, num_points + 1, endpoint=False)
    samples = np.sort(samples).astype(int)[1:]
    
    polynom_degree = num_points + 2 if not polynom_degree else polynom_degree

    # creates the wall of the and the vein as the two lines it creates randomly N points shifts them randomly up and down 
    points_up = [[0, l_u_y]]
    points_down = [[0, l_d_y]]
    for x_coord_up in samples:
        while True:
            center_position = (l_center+r_center)//2
            average_width = (l_width+r_width)//2
            center = round(random.triangular(center_position - major_shift, center_position + major_shift, center_position))
            x_coord_down = round(random.triangular(x_coord_up-minor_shift, x_coord_up+minor_shift, x_coord_up))
            y_coord_up = center - average_width//2 + round(random.triangular(-minor_shift, +minor_shift, 0))
            y_coord_down = center + average_width//2 + round(random.triangular(-minor_shift, +minor_shift, 0))
            if y_coord_up < 256 and y_coord_down > 0 and y_coord_down - y_coord_up > 80:
                points_up.append([x_coord_up, y_coord_up])
                points_down.append([x_coord_down, y_coord_down])
                break
           

    points_up.append([255, r_u_y])
    points_down.append([255, r_d_y])
    
    #create lines by interpolating the points
    vein_up_line = np.array(points_up, np.int32)
    coefficients_up = np.polyfit(vein_up_line[:,0], vein_up_line[:,1],polynom_degree)
    up_interpolated_x = np.linspace(0, 256, num_points*points_multiplier_for_interpolation, endpoint=True)
    up_interpolated_y = np.polyval(coefficients_up, up_interpolated_x)
    vein_up_line = np.array([np.column_stack((up_interpolated_x, up_interpolated_y))], np.int32)
    
    vein_down_line = np.array(points_down, np.int32)
    coefficients_down = np.polyfit(vein_down_line[:,0], vein_down_line[:,1],polynom_degree)
    down_interpolated_x = np.linspace(0, 256, num_points*points_multiplier_for_interpolation, endpoint=True)
    down_interpolated_y = np.polyval(coefficients_down, down_interpolated_x)
    vein_down_line = np.array([np.column_stack((down_interpolated_x, down_interpolated_y))], np.int32)
    
    
    # fill lumen
    vertices = np.concatenate((vein_up_line[0], np.flip(vein_down_line[0], axis=0)), axis=0)
    color = (0, 0, 255)  # blue color
    cv2.fillPoly(base_image, [vertices], color)
    
    # add plaque
    plaque_length = randint(round(down_interpolated_x.size/3), round(down_interpolated_x.size/2))
    start = randint(0, down_interpolated_x.size - plaque_length)
    end = start + plaque_length
    add_values = generate_random_heights_of_plaque(plaque_length, randint(10,30))
    plaque_x = down_interpolated_x[start:end]
    plaque_y = []
    for i in range(plaque_length):
        p_y = down_interpolated_y[start+i] - add_values[i]
        plaque_y.append(round(p_y))
    plaque_vertices = np.concatenate((vein_down_line[0][start:end], np.flip(np.column_stack((plaque_x, plaque_y)), axis=0)), axis=0).astype(np.int32)

    # fill lumen
    color = (0, 255, 0)  # blue color
    cv2.fillPoly(base_image, [plaque_vertices], color)
    cv2.polylines(base_image, [vein_up_line - [0, 3]], False, (255,0,0), thickness=6)
    cv2.polylines(base_image, [vein_down_line + [0, 3]], False, (255,0,0), thickness=6)

    if(deform):
        # This is the grid distortion and elastic transformation
        base_image = Grid_distortion(base_image)
        base_image = Elastic_transform(base_image)


        for x in range(256):
            for y in range(256):
                if not np.array_equal(base_image[x, y], [0., 255., 0.]) and not np.array_equal(base_image[x, y], [255., 0., 0.]) \
                    and not np.array_equal(base_image[x, y], [0., 0., 255.]) and not np.array_equal(base_image[x, y], [0., 0., 0.]):
                    if np.dot(base_image[x, y], [0., 255., 0.])/(np.linalg.norm(base_image[x, y]) *np.linalg.norm([0., 255., 0.]) ) >= 0.65:
                        base_image[x, y, :] = [0, 255, 0]
                    elif np.dot(base_image[x, y], [0., 0., 255.])/(np.linalg.norm(base_image[x, y]) *np.linalg.norm([0., 0., 255.]) ) >= 0.65:
                        base_image[x, y, :] = [0, 0, 255]
                    else:
                        base_image[x, y, :] = [255, 0, 0]

    if meassure:
        meassurements = compute_longitudal_params(base_image)
    else:
        meassurements = []
    if show:
        plt.imshow(base_image)
        plt.show()
    return base_image.astype(np.uint8), meassurements