import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import *
import sys

A = None


def a():
    print("(a)")

    fig = plt.figure()

    arr = np.array(A.reshape(-1))
    
    fig.gca().plot(range(len(arr)), np.sort(arr)[::-1], label="magnitude")

    fig.gca().legend()

    fig.gca().set_title("4a) Magnitude of values of A")
    fig.gca().set_ylabel("Magnitude")

    fig.savefig("4a.png")
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

    fig.gca().set_title("4b) Histogram of the Magnitudes in A")

    fig.savefig("4b.png")
    #plt.show()

def c():
    print("(c)")

    X = np.array(A[50:100,0:50])
    np.save("outputXPS0Q1.npy", X)

    fig = plt.figure()

    fig.gca().imshow(X, interpolation="nearest")

    fig.gca().set_title("4c) Bottom Left Quadrant of A")

    fig.savefig("4c.png")

def d():
    print("(d)")

    Y = A - np.mean(A)
    np.save("outputYPS0Q1.npy", Y)

    print("Note: Since A is sampled from random normal distribution, and as seen from part b), the mean is very close to 0, and so there is very little difference between A and Y")


    fig = plt.figure()

    fig.gca().imshow(Y, interpolation="nearest")
    fig.gca().set_title("4d) A shifted by mean")

    fig.savefig("4d.png")

def e():

    print("(e)")

    cmap = ListedColormap([[0, 0, 0], [1, 0, 0]])

    Z = A > np.mean(A).astype(np.uint8) 
    np.save("outputZPS0Q1.npy", Z)

    fig = plt.figure()

    fig.gca().imshow(Z, interpolation="nearest", cmap=cmap)
    fig.gca().set_title("4e) Red if A > t")

    fig.savefig("outputZPS0Q1.png")

if __name__ == "__main__":
    # load A from source
    A = np.load("inputAPS0Q1.npy")
    
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
