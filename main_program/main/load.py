"The goal of this file is to save and load the image we will give it."

import glob
import cv2

def load_image(path_image):
    """loads an image from the specified path.

    Args:
        path_nbr (string): string to image

    Returns:
        numPy Array of the image
    """
    return cv2.imread(path_image)
    

def load_multiple_image(path_image):
    filenames = glob.glob(path_image)
    images = [cv2.imread(img) for img in filenames]
    for img in images:
        load_image(img)