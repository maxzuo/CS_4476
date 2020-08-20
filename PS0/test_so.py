import numpy as np
import timeit

list_time = timeit.timeit(stmt="a = [1, 2, 3, 4, 5]", number=1000000)
ndarray_time = timeit.timeit(stmt=lambda: "a = np.array([1, 2, 3, 4, 5])", setup="import numpy as np", number=1000000)