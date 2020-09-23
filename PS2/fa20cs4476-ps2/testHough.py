import numpy as np
import matplotlib.image as Image
import matplotlib.pyplot as plt
from submissionDetectCircles import *

if __name__ == "__main__":
    radius = 109.5
    image = Image.imread("../jupiter.jpg")

    edges, votes, centers = detect_circles(image, radius, useGradient=True, sigma=1, return_votes=True)

    # print(np.sum(edges), np.max(votes))
    print(centers)

    plt.figure().gca().imshow(edges.astype(np.uint8))
    plt.figure().gca().imshow(votes)

    plt.show()

    plt.imshow(image)
    for center in centers:
        draw_circle = plt.Circle(center[::-1], radius, fill=False, color='g')

        plt.gcf().gca().add_artist(draw_circle)
    plt.title('Circle')

    plt.show()


    radius = 5
    image = Image.imread("../egg.jpg")

    edges, votes, centers = detect_circles(image, radius, useGradient=True, sigma=2, return_votes=True, ratio=0.7)

    # print(np.sum(edges), np.max(votes))
    print(centers)

    plt.figure().gca().imshow(edges.astype(np.uint8))
    plt.figure().gca().imshow(votes)

    plt.show()

    plt.imshow(image)
    for center in centers:
        draw_circle = plt.Circle(center[::-1], radius, fill=False, color='r')

        plt.gcf().gca().add_artist(draw_circle)
    plt.title('Circle')

    plt.show()
