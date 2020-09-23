
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# Edit KMeans.ipynb instead.
import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2hsv, hsv2rgb
from typing import Tuple

def quantize_rgb(img: np.ndarray, k: int) -> np.ndarray:
    """
    Compute the k-means clusters for the input image in RGB space, and return
    an image where each pixel is replaced by the nearest cluster's average RGB
    value.

    Inputs:
        img: Input RGB image with shape H x W x 3 and dtype "uint8"
        k: The number of clusters to use

    Output:
        An RGB image with shape H x W x 3 and dtype "uint8"
    """
    quantized_img = np.zeros_like(img)

    ##########################################################################
    # TODO: Perform k-means clustering and return an image where each pixel  #
    # is assigned the value of the nearest clusters RGB values.              #
    ##########################################################################

    quantized_img = np.array(img)
    pixels = quantized_img.reshape((-1, 3))

    kM = KMeans(n_clusters=k, random_state=101).fit(pixels.astype(np.float))
    meanColors, labels = kM.cluster_centers_, kM.labels_

    for i in range(len(labels)):
        color = meanColors[labels[i]]
        pixels[i] = color.astype(np.uint8)

    ##########################################################################
    ##########################################################################

    return quantized_img

def quantize_hsv(img: np.ndarray, k: int) -> np.ndarray:
    """
    Compute the k-means clusters for the input image in the hue dimension of the
    HSV space. Replace the hue values with the nearest cluster's hue value. Finally,
    convert the image back to RGB.

    Inputs:
        img: Input RGB image with shape H x W x 3 and dtype "uint8"
        k: The number of clusters to use

    Output:
        An RGB image with shape H x W x 3 and dtype "uint8"
    """

    ##########################################################################
    # TODO: Convert the image to HSV. Perform k-means clustering in hue      #
    # space. Replace the hue values in the image with the cluster centers.   #
    # Convert the image back to RGB.                                         #
    ##########################################################################

    quantized_img = np.array(rgb2hsv(img))
    pixels = quantized_img.reshape((-1, 3))[:, 0].reshape((-1, 1))

    kM = KMeans(n_clusters=k, random_state=101).fit(pixels.astype(np.float))
    meanHues, labels = kM.cluster_centers_, kM.labels_

    for i in range(len(labels)):
        hue = meanHues[labels[i]]
        pixels[i] = hue

    quantized_img = (255 * hsv2rgb(quantized_img)).astype(np.uint8)

    ##########################################################################
    ##########################################################################

    return quantized_img

def compute_quantization_error(img: np.ndarray, quantized_img: np.ndarray) -> int:
    """
    Compute the sum of squared error between the two input images.

    Inputs:
        img: Input RGB image with shape H x W x 3 and dtype "uint8"
        quantized_img: Quantized RGB image with shape H x W x 3 and dtype "uint8"

    Output:

    """
    error = 0

    ##########################################################################
    # TODO: Compute the sum of squared error.                                #
    ##########################################################################



    ##########################################################################
    ##########################################################################

    return np.sum(np.square(img.astype(np.float32) - quantized_img.astype(np.float32)))

def get_hue_histograms(img: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the histogram values two ways: equally spaced and clustered.

    Inputs:
        img: Input RGB image with shape H x W x 3 and dtype "uint8"
        k: The number of clusters to use

    Output:
        hist_equal: The values for an equally spaced histogram
        hist_clustered: The values for a histogram of the cluster assignments
    """
    hist_equal = np.zeros((k,), dtype=np.int64)
    hist_clustered = np.zeros((k,), dtype=np.int64)

    ##########################################################################
    # TODO: Convert the image to HSV. Calculate a k-bin histogram for the    #
    # hue dimension. Calculate the k-means clustering of the hue space.      #
    # Calculate the histogram values for the cluster assignments.            #
    ##########################################################################
#     pixels = (rgb2hsv(img) * 255).astype(np.uint8).reshape((-1,3))[:,0].reshape((-1, 1))
    pixels = (rgb2hsv(img)[:,:,0]).reshape((-1,1))

#     _max, _min = np.max(pixels), np.min(pixels)
#     _diff = _max - _min
#     bins = [(_diff) / k * (i) + _min + 1 for i in range(k+1)]
#     bins[0] = 0
#     bins[-1] = 256
    hist_equal, bins = np.histogram(pixels.flatten(), bins=k)

#     print(pixels)
#     equal_pix = ((pixels - np.min(pixels)).astype(np.float32) *  (k - 1) / (np.max(pixels) - np.min(pixels))).astype(np.uint8)
#     print(equal_pix.reshape(-1))
# #     hist_equal[equal_pix.reshape(-1)] += 1
#     for pix in equal_pix: hist_equal[pix] += 1


    kM = KMeans(n_clusters=k, random_state=101).fit(pixels)
    meanHues, labels = kM.cluster_centers_, kM.labels_

    for i in range(len(labels)): hist_clustered[labels[i]] += 1


    ##########################################################################
    ##########################################################################

    return hist_equal, hist_clustered