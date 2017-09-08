import numpy as np

def decode_class(vec, class_names):  # generates a number from the one-hot vector
    return int(np.argmax(vec))
