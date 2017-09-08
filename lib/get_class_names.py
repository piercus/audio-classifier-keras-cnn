import os
from os.path import isfile

def get_class_names(path="Preproc/"):  # class names are subdirectory names in Preproc/ directory
    class_names = os.listdir(path)
    return class_names
