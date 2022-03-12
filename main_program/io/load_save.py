"The goal of this file is to save and load the image we will give it."

import json
import cv2
import os


def load_image(path_nbr):
    """loads an image from the specified path.

    Args:
        path_nbr (string): string to image

    Returns:
        numPy Array of the image
    """
    return cv2.imread(path_nbr)