"The goal of this file is to save and load the image we will give it."

import json
import cv2
import os


def load_image(path_image):
    """loads an image from the specified path.

    Args:
        path_nbr (string): string to image

    Returns:
        numPy Array of the image
    """
    return cv2.imread(path_image)
    

def load_image_data(json_path):
    """loads a json file and an image
    Args:
        json_path (string): path of the json file.
    Returns:
        dictionnary: image and json file into a dictionary.
    """
    image_path = json_path.split(".json")[0] + ".png"
    image_data = load_json(json_path)
    image_data["image"] = load_image(image_path)
    return image_data