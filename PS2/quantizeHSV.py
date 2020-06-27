import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
from scipy.cluster.vq import kmeans2

def quantizeHSV(origImg, k):
    outputImg = np.array(rgb2hsv(origImg))
    pixels = outputImg.reshape((-1, 3))[:, 0]
    
    meanHues, labels = kmeans2(pixels.astype(np.float), k, minit='points', iter=1_000)

    for i in range(len(labels)):
        hue = meanHues[labels[i]]
        pixels[i] = hue
    
    return (255 * hsv2rgb(outputImg)).astype(np.uint8), meanHues
