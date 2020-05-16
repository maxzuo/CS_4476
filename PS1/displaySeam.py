import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt

def displaySeam(im, seam, type):
    figure = plt.figure()

    figure.gca().imshow(im)

    if type.lower() == "horizontal":
        figure.gca().plot(np.arange(seam.shape[0]), seam)
    elif type.lower() == "vertical":
        figure.gca().plot(seam, np.arange(seam.shape[0]))

    return figure