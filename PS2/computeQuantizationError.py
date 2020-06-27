import numpy as np

def computeQuantizationError(origImg, quantizedImg):
    return np.sum(np.square(origImg - quantizedImg))