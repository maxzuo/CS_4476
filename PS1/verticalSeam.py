import numpy as np

def find_optimal_vertical_seam(cumulativeEnergyMap):
    # which way do the column indices go?
    seam = np.zeros((cumulativeEnergyMap.shape[0],), dtype=np.int)
    
    seam[-1] = np.argmin(cumulativeEnergyMap[-1])
    for i in reversed(range(cumulativeEnergyMap.shape[0] - 1)):
        last = seam[i+1]
        seam[i] = max(np.argmin(cumulativeEnergyMap[i, max(last-1, 0):last+2]) + last - 1, 0)
    return seam