import numpy as np
from skimage.feature import canny as canny
from skimage.color import rgb2gray as rgb2gray

np.random.seed(101)

def detect_circles(im, radius, useGradient=False, sigma=None, qStep=100, ratio=0.85, return_votes=False):

    base = rgb2gray(im)
    X, Y = base.shape

    if sigma:
        edges = canny(base, sigma=sigma)
    else:
        edges = canny(base, sigma=1 + ((X + Y) // 100))
    # print(edges)

    if useGradient:
        dx, dy = np.gradient(base)
    # Determine vote space quantization?

    votes = np.zeros(base.shape, dtype=np.int)


    # calculate thetas
    # T = [None] * qStep
    # theta = 0
    # step = 2 * np.pi / qStep
    # for _ in range(0, qStep):
    #     theta += step
    #     T[_] = (int(radius * np.cos(theta)), int(radius * np.sin(theta)))
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / qStep)
    T = np.vstack([radius * np.cos(thetas), radius * np.sin(thetas)]).T
    # Make votes (Hough Space)


    for i, j in np.argwhere(edges != 0):
        if not edges[i][j]:
            continue

        if useGradient:
            theta = np.arctan2(dy[i][j], dx[i][j])

            a = (radius * np.cos(theta)).astype(np.int) + i
            b = (radius * np.sin(theta)).astype(np.int) + j

            if not (a < 0 or a >= X or b < 0 or b >= Y):
                votes[a][b] += 1

            a = (radius * -np.cos(theta)).astype(np.int) + i
            b = (radius * -np.sin(theta)).astype(np.int) + j

            if not (a < 0 or a >= X or b < 0 or b >= Y):
                votes[a][b] += 1

        else:
            for x, y in T:
                a, b = x + i, y + j
                if (a < 0 or a >= X): continue
                if (b < 0 or b >= Y): continue

                votes[a][b] += 1
    centers = np.argwhere(votes > ratio * np.max(votes))
    if return_votes:
        return edges, votes, centers
    else:
        return centers



    # Quantize vote?

    # select