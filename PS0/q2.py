import numpy as np


def random_dice(N): return (6 * np.random.rand(N) + 1).astype(int) # results stores the faces (i.e. 1, 2, 3, 4, 5, 6) rolled from N different trials

def reshape_vector(y=np.array([1, 2, 3, 4, 5, 6])): return y.reshape((-1, 2))

def max_value(z=np.array([[1,2],[3,4],[5,6]])): return np.where(z == np.max(z))

def count_ones(v=np.array([1, 8, 8, 2, 1, 3, 9, 8])): return np.sum(v == 1)