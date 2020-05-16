import numpy as np


def rgb2gray(image):
    return image[...,:] @ [0.2989, 0.5870, 0.1140]

def energy_image(im):
    if im.shape[-1] == 3:
        im = rgb2gray(im)
    
    # WHICH ENERGY FUNCTION DO WE USE? The one on the slides or the one in the paper???
    # g = np.sum(np.abs(np.gradient(im)), axis=0)
    
    g = np.sqrt(np.sum(np.square(np.gradient(im)), axis=0))

    return g

    # return rgb2gray(g)