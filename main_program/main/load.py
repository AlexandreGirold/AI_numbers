"The goal of this file is to save and load the image we will give it."

import glob
import cv2
import os

path_folder = r"C:\Users\sacha\Desktop\AI_numbers\main_program\images"

def load_image(image_path):
    """Loads an image from a path.
    Args:
        image_path (string): path of the image to load.
    Returns:
        np.array: the image.
    """
    return cv2.imread(image_path)

def load_multiple_images(dirpath):
    """load dataset (png from a folder).
        *ps glob.glob gives a list of path.
    Args:
        dirpath (string): path of a directory which contains json and png.
    Returns:
        list[dict]: list of dictionnary containing image and json.
    """

    for filepath in glob.glob(dirpath + os.sep + "*.png"):
        yield load_image(filepath)
    
def get_answer(dirpath):
    answer = []
    for filepath in glob.glob(dirpath + os.sep + "*.png"):
        numb = os.path.basename(filepath).split("_")[0]
        answer.append(numb)
    return answer