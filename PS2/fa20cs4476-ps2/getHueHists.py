import numpy as np
from skimage.color import rgb2hsv
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt

def getHueHists(img, k):
    pixels = (rgb2hsv(img) * 255).astype(np.uint8).reshape((-1,3))[:,0]
    
    # even-spaced bins
    bins = np.histogram(pixels, bins=k)[1]

    histEqual = plt.figure()

    histEqual.gca().hist(pixels, bins=bins)
    histEqual.gca().set_xticks(bins)
    histEqual.gca().set_title("Equally Spaced Hues Histogram k=%d" % k)

    histEqual.gca().set_xlabel("Hue (0-255)")

    # quantized bins
    meanHues, labels = kmeans2(pixels.astype(np.float), k, minit='points', iter=10_000)

    reorder = np.argsort(meanHues)
    for i in range(len(labels)): labels[i] = np.argwhere(reorder == labels[i])
    meanHues = meanHues[reorder]

    histClustered = plt.figure()

    bins = np.arange(k+1)
    
    histClustered.gca().hist(labels, bins=bins)
    histClustered.gca().tick_params(axis="x", which="minor", length=0)
    histClustered.gca().set_xticks(bins)
    histClustered.gca().set_xticklabels('')
    histClustered.gca().set_xticks(bins + 0.5, minor=True)
    histClustered.gca().set_xticklabels(map(lambda s: "%.2f" % s, meanHues), minor=True)
    histClustered.gca().set_title("Quantized Hues Histogram k=%d" % k)

    histClustered.gca().set_xlabel("Hue (0-255)")

    return histEqual, histClustered