import numpy as np
from skimage.feature import canny as canny
from skimage.color import rgb2gray as rgb2gray
from scipy import signal, ndimage
from sklearn.cluster import MeanShift
from collections import Counter

np.random.seed(101)

def detect_circles(img, radius, useGradient=False, sigma=None, qStep=100, thresh=0.85, min_thresh=3.5, return_votes=False, q_dim=1, mean_shift=False):
    """
    Detect and return circles from an RGB (H X W X 3) image using the Hough Transform

    Inputs:
        img:            H x W x 3 RGB image of datatype uint8
        radius:         target radius of the circle to detect
        useGradient:    whether or not to use the gradient to help detect circles
        sigma:          sigma used for canny edge detector
        qStep:          number of angles for theta to use in voting if not using useGradient
        thresh:         float in the range (0,1] determining the percentage of the max votes required
                        to also be considered a circle
        min_thresh:     float greater than 0, determining the multiplying factor greater than the median
                        that is required to be considered a center.
        return_votes:   if set to true, will also return edges and accumulator array
        q_dim:   default=1, setting this higher will but votes into (q_dim x q_dim) pixel groups.
    """
    base = rgb2gray(img)
    X, Y = base.shape

    if sigma:
        edges = canny(base, sigma=sigma)
    else:
        edges = canny(base, sigma=1 + ((X + Y) // 100))
    # print(edges)

    if useGradient:
        dx, dy = np.gradient(base)

    # initialize vote space
    votes = np.zeros(base.shape, dtype=np.int)


    # pre-calculate radial changes
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / qStep)
    T = np.vstack([radius * np.cos(thetas), radius * np.sin(thetas)]).T.astype(np.int)

    # Make votes (Hough Space)


    for i, j in np.argwhere(edges != 0):

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

    if q_dim != 1:
        votes = signal.convolve2d(votes, np.ones((q_dim, q_dim)), 'same')
        votes = ndimage.maximum_filter(votes, size=q_dim)[0::q_dim, 0::q_dim]
    if mean_shift:
        median = np.median(votes)
        vMax = np.max(votes)
        ms = MeanShift(bandwidth=radius//2).fit(np.argwhere(np.multiply((votes > min_thresh * np.median(votes)), (votes > thresh * np.max(votes)))))
        centers = ms.cluster_centers_.astype(np.int) * q_dim
    else:
        centers = np.argwhere(np.multiply((votes > min_thresh * np.median(votes)), (votes > thresh * np.max(votes)))) * q_dim
    if return_votes:
        return edges, votes, centers
    else:
        return centers


############################
#
# Extra Credit
#
############################

def variable_radius_hough_circle(img, min_radius, max_radius, useGradient=False, sigma=None, qStep=100, thresh=0.85, min_thresh=3.5, return_votes=False, q_dim=1):
    """
    Detect and return circles from an RGB (H X W X 3) image using the Hough Transform

    Inputs:
        img:            H x W x 3 RGB image of datatype uint8
        radius:         target radius of the circle to detect
        useGradient:    whether or not to use the gradient to help detect circles
        sigma:          sigma used for canny edge detector
        qStep:          number of angles for theta to use in voting if not using useGradient
        thresh:         float in the range (0,1] determining the percentage of the max votes required
                        to also be considered a circle
        min_thresh:     float greater than 0, determining the multiplying factor greater than the m*dian
                        that is required to be considered a center.
        return_votes:   if set to true, will also return edges and accumulator array
        q_dim:   default=1, setting this higher will but votes into (q_dim x q_dim) pixel groups.
    Returns:
        centers:        list of centers (x, y, radius)
    """

    base = rgb2gray(img)

    if sigma:
        edges = canny(base, sigma=sigma)
    else:
        edges = canny(base, sigma=1 + ((X + Y) // 100))
    # print(edges)

    if useGradient:
        dx, dy = np.gradient(base)
    else:
        thetas = np.arange(0, 2 * np.pi, 2 * np.pi / qStep)
        T = np.vstack([np.cos(thetas), np.sin(thetas), np.ones(thetas.shape)]).T.astype(np.int)
        T = np.vstack([T * (i - i % q_dim) for i in np.arange(min_radius, max_radius, q_dim)])

    # sparse vote space
    votes = []

    # We will be using a Counter object later to keep track of the circle votes
    vote_counter = None

    for i, j in np.argwhere(edges != 0):

        if useGradient:
            temp_votes = []
            for radius in np.arange(min_radius, max_radius, q_dim):
                radius -= radius % q_dim
                theta = np.arctan2(dy[i][j], dx[i][j])

                a = (radius * np.cos(theta)).astype(np.int) + i
                b = (radius * np.sin(theta)).astype(np.int) + j

                a = a - a % q_dim
                b = b - b % q_dim

                votes.append((a,b,radius))


                # other direction
                a = (radius * -np.cos(theta)).astype(np.int) + i
                b = (radius * -np.sin(theta)).astype(np.int) + j

                a = a - a % q_dim
                b = b - b % q_dim

                votes.append((a,b,radius))

        else:
            for x, y, r in T:
                a, b = x + i, y + j

                a = a - a % q_dim
                b = b - b % q_dim

                votes.append((a,b, r))

    vote_counter = Counter(votes)
    vMean = np.mean(list(vote_counter.values()))
    vMax = vote_counter.most_common(1)[0][1]

    # readjust votes
    votes = []
    max_thresh = max(vMean * min_thresh, vMax * thresh)
    for point, vote in vote_counter.items():
        if vote > max_thresh:
            votes.append(point)
            for i in range(int(max_thresh), vote, 2):
                votes.append(point)

    votes = np.array(votes)
    print(votes.shape)
    ms = MeanShift(bandwidth=min_radius//2, n_jobs=-1).fit(votes)
    return ms.cluster_centers_.astype(np.int)


