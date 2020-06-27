import numpy as np
from skimage.feature import canny as canny
from skimage.color import rgb2gray as rgb2gray

def detectCircles(im, radius, useGradient=False, sigma=None, qStep=100):

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
    T = [None] * qStep
    theta = 0
    step = 2 * np.pi / 100
    for _ in range(0, qStep):
        theta += step
        T[_] = (int(radius * np.cos(theta)), int(radius * np.sin(theta)))
    # Make votes (Hough Space)
    
    
    for i in range(X):
        for j in range(Y):
            if not edges[i][j]:
                continue
            
            if useGradient:
                s = dy[i][j] / dx[i][j]
                theta = np.arctan(s)
                
                a = (radius * np.cos(theta)).astype(np.int) + i
                b = (radius * np.sin(theta)).astype(np.int) + j

                if not (a < 0 or a >= X or b < 0 or b >= Y):
                    votes[a][b] += 1

                a = (radius * np.cos(theta)).astype(np.int) + i
                b = (radius * np.sin(theta)).astype(np.int) + j

                if not (a < 0 or a >= X or b < 0 or b >= Y):
                    votes[a][b] += 1

            else:
                for x, y in T:
                    a, b = x + i, y + j
                    if (a < 0 or a >= X): continue
                    if (b < 0 or b >= Y): continue

                    votes[a][b] += 1

    return edges, votes



    # Quantize vote?

    # select 