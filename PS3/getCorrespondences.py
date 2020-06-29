import numpy as np
import matplotlib.pyplot as plt


def getCorrespondences(image, image1):
    coordinates = []
    coordinates1 = []
    def onclick(c, fig):
        def m(e):
            c.append([e.xdata, e.ydata])
            print(coordinates, coordinates1)
            # if len(coordinates) == 4: plt.close(fig)
        return m

    fig = plt.figure()
    fig.gca().imshow(image)
    fig.canvas.mpl_connect('button_press_event', onclick(coordinates, fig))


    fig1 = plt.figure()
    fig1.gca().imshow(image1)
    fig1.canvas.mpl_connect('button_press_event', onclick(coordinates1, fig1))

    plt.show()

    return np.asarray(coordinates), np.asarray(coordinates1)


if __name__ == "__main__":
    image = plt.imread("crop1.jpg")
    image1 = plt.imread("crop2.jpg")

    print(getCorrespondences(image, image1))

