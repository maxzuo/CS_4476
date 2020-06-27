import quantizeRGB, quantizeHSV, computeQuantizationError, getHueHists, colorQuantizeMain
import matplotlib.image as Image
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, hsv2rgb

def test_qRGB(img, k):
    outputImage, meanColors = quantizeRGB.quantizeRGB(img, k)

    plt.imshow(outputImage)
    plt.show()

def test_qHSV(img, k):
    outputImage, meanHues = quantizeHSV.quantizeHSV(img, k)

    plt.imshow(outputImage)
    plt.show()

def test_qError(img):
    outputImage, _ = quantizeRGB.quantizeRGB(img, 1)
    ssd = computeQuantizationError.computeQuantizationError(img, outputImage)
    print("k: %d\t ssd: %d" % (1, ssd))
    for i in range(5, 51, 5):
        outputImage, _ = quantizeRGB.quantizeRGB(img, i)
        ssd = computeQuantizationError.computeQuantizationError(img, outputImage)
        print("k: %d\t ssd: %d" % (i, ssd))


def test_hist(img):

    e, q = getHueHists.getHueHists(img, 5)

    plt.show()

if __name__ == "__main__":

    img = Image.imread("fish.jpg")

    # test_qRGB(img, 10)

    # test_qHSV(img, 5)

    # test_qError(img)

    test_hist(img)