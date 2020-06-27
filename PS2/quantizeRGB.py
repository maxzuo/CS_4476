import numpy as np
from scipy.cluster.vq import kmeans2

def quantizeRGB(origImg, k):
    outputImg = np.array(origImg)
    pixels = outputImg.reshape((-1, 3))
    
    meanColors, labels = kmeans2(pixels.astype(np.float), k,  minit='points', iter=1_000)

    for i in range(len(labels)):
        color = meanColors[labels[i]]
        pixels[i] = color.astype(np.uint8)
    
    return outputImg, meanColors