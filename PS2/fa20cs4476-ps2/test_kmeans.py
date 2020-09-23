from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt



qStep = 40
qStep2 = 90



for _ in range(5):
    T = [None] * (qStep + qStep2)
    theta = 0
    step = 2 * np.pi / qStep
    for _ in range(0, qStep):
        theta += step
        T[_] = (int(40 * np.cos(theta)), int(40 * np.sin(theta)))

    step = 2 * np.pi / qStep2
    theta = 0
    for _ in range(0, qStep2):
        theta += step
        T[_ + qStep] = (int(80 * np.cos(theta)), int(80 * np.sin(theta)))

    T = np.array(T)
    kM = KMeans(n_clusters=2, init=np.array([[0, 0], [0, 3000]])).fit(T.astype(np.float))
    center, labels = kM.cluster_centers_, kM.labels_


    plt.scatter(T[:,0], T[:,1], c=labels)
    plt.scatter(center[:,0], center[:,1])
    plt.show()
