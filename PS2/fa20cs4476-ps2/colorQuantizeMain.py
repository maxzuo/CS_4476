from submissionKMeans import quantize_rgb as quantizeRGB, quantize_hsv as quantizeHSV, compute_quantization_error as computeQuantizationError, get_hue_histograms as getHueHists
import matplotlib.image as Image
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, hsv2rgb

def test_qRGB(img, k):
    outputImage = quantizeRGB(img, k)

    Image.imsave(f"fish_qRGB_{k}.png", outputImage)

    # plt.imshow(outputImage)
    # plt.show()

def test_qHSV(img, k):
    outputImage = quantizeHSV(img, k)

    Image.imsave(f"fish_qHSV_{k}.png", outputImage)

    # plt.imshow(outputImage)
    # plt.show()
    # plt.imshow(img)
    # plt.show()

def test_qError(img):
    # outputImage = quantizeRGB(img, 1)
    # ssd = computeQuantizationError(img, outputImage)
    # print("k: %d\t ssd: %d" % (1, ssd))
    # for i in range(5, 51, 5):
    #     outputImage = quantizeRGB(img, i)
    #     ssd = computeQuantizationError(img, outputImage)
    #     print("k: %d\t ssd: %d" % (i, ssd))

    outputImage = quantizeHSV(img, 1)
    ssd = computeQuantizationError(img, outputImage)
    print("k: %d\t ssd: %d" % (1, ssd))
    for i in range(5, 51, 5):
        outputImage = quantizeHSV(img, i)
        ssd = computeQuantizationError(img, outputImage)
        print("k: %d\t ssd: %d" % (i, ssd))

def plot_hist(h, title, filename, aspect=None):
    plt.hist(range(len(h)), range(0,len(h) + 1), weights=h)
    plt.gca().tick_params(axis="x", which="minor", length=0)
    plt.gca().tick_params(axis="x", which="major", length=0)
    plt.gca().set_xticklabels('')
    plt.gca().set_title(title)

    # if aspect:
    #     plt.gca().set_aspect(aspect * max(h) / len(h))
    plt.savefig(filename)
    plt.show()
    plt.clf()

def test_hist(img):

    e, c = getHueHists(img, 5)


    plot_hist(e, "Equally Spaced Hues Histogram, k=5", "fish_evenHues.png")
    plot_hist(c, "Clustered/Quantized Hues Histogram, k=5", "fish_quantizedHues.png")

    e, c = getHueHists(img, 50)

    plot_hist(e, "Equally Spaced Hues Histogram, k=50", "fish_evenHues50.png", 5)
    plot_hist(c, "Clustered/Quantized Hues Histogram, k=50", "fish_quantizedHues50.png", 5)

    # plt.show()

if __name__ == "__main__":

    img = Image.imread("fish.jpg")

    # test_qRGB(img, 50)
    # test_qRGB(img, 10)
    # test_qRGB(img, 5)

    # test_qHSV(img, 50)
    # test_qHSV(img, 10)
    # test_qHSV(img, 5)

    test_qError(img)

    # test_hist(img)