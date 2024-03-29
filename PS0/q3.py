import numpy as np
from matplotlib import pyplot as plt
import imageio as Image
import sys

A = None


def a():
    print("(a)")

    fig = plt.figure()

    arr = np.array(A.reshape((-1,1)))
    
    arr_sorted = np.sort(arr, axis=0)[::-1,:]

    indices = np.arange(arr.shape[0], step=999)
    # print(indices, arr.shape)

    fig.gca().set_title("3a) Magnitude of values of A")
    fig.gca().set_xlabel("Magnitude")
    fig.gca().axes.yaxis.set_visible(False)
    fig.gca().axes.set_xticks(indices)
    fig.gca().axes.set_xticklabels(list(map(lambda s: "%.2f" % s, arr_sorted[indices,0])))

    fig.gca().imshow(np.tile(arr_sorted, 3000).T, cmap="gray")
    # fig.colorbar(arr_sorted, cmap='gray')

    fig.savefig("3a.png")
    #plt.show()

def b():
    print("(b)")

    fig = plt.figure(figsize=(10, 5))

    arr = np.array(A.reshape(-1))
    bins = [-5 + i/2 for i in range(21)]
    
    fig.gca().hist(arr,bins=bins)
    
    fig.gca().set_xticks(bins)
    fig.gca().set_xlabel("Magnitude")

    fig.gca().set_ylabel("Number of occurrences in A")

    fig.gca().set_title("3b) Histogram of the Magnitudes in A")

    fig.savefig("3b.png")
    #plt.show()

def c():
    print("(c)")

    X = np.array(A[50:100,0:50])
    np.save("q3-output-x.npy", X)

    fig = plt.figure()

    fig.gca().imshow(X, interpolation="nearest")

    fig.gca().set_title("3c) Bottom Left Quadrant of A")

    fig.savefig("3c.png")

def d():
    print("(d)")

    Y = A - np.mean(A)
    np.save("q3-output-y.npy", Y)

    print("Note: Since A is sampled from random normal distribution, and as seen from part b), the mean is very close to 0, and so there is very little difference between A and Y")


    fig = plt.figure()

    fig.gca().imshow(Y, interpolation="nearest")
    fig.gca().set_title("3d) A shifted by mean")

    fig.savefig("3d.png")

def e():

    print("(e)")

    Z = np.zeros((*A.shape, 3), dtype=np.uint8)
    
    Z[:,:,0] = (A > np.mean(A)) * 255
    np.save("q3-output-Z.npy", Z)

    fig = plt.figure()

    fig.gca().imshow(Z, interpolation="nearest")
    fig.gca().set_title("3e) Red if A > t")

    fig.savefig("3e.png")

    Image.imsave("q3-output-z.png", Z)
    

if __name__ == "__main__":
    # load A from source
    A = np.load("q3-input.npy")
    
    if A is None or A.shape != (100, 100):
        print("A is not loaded correctly")
        sys.exit(1)

    # Question 4, part (a)
    a()

    # Question 4, part (b)
    b()

    # Question 4, part (c)
    c()

    # Question 4, part (d)
    d()

    # Question 4, part (e)
    e()

    plt.show()
