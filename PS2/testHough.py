import numpy as np
import matplotlib.image as Image
import matplotlib.pyplot as plt
from detectCircles import *

if __name__ == "__main__":
    image = Image.imread("egg.jpg")

    edges, votes = detectCircles(image, 5, useGradient=True, sigma=1)

    # print(np.sum(edges), np.max(votes))

    plt.figure().gca().imshow(edges.astype(np.uint8))
    plt.figure().gca().imshow(votes)

    plt.show()