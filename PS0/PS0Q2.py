import numpy as np
import matplotlib.pyplot as plt

image = None

def swap_red_green(image):
    swapped = np.array(image)

    swapped[:, :, 0], swapped[:, :, 1] = swapped[:, :, 1], swapped[:, :, 0]

    return swapped

def rgb2gray(image):
    #return np.mean(rgb, -1)
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

def grayscale_negative(image):
    return 255 - image

if __name__ == "__main__":
    
    fig = plt.figure()
    image = fig.gca().imread("inputPS0Q2.jpg")
