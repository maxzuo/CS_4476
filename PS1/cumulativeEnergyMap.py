import numpy as np

def cumulative_minimum_energy_map(energyImage, seamDirection):
    energyMap = np.array(energyImage)

    if seamDirection.lower().strip() == "horizontal":
        energyMap.T[0] = energyImage.T[0]
        for j in range(1, energyMap.shape[1]):
            for i in range(0, energyMap.shape[0]):
                energyMap[i][j] += np.min(energyMap[max(i-1,0):i+2, j-1])
    elif seamDirection.lower().strip() == "vertical":
        energyMap[0] = energyImage[0]
        for i in range(1, energyMap.shape[0]):
            for j in range(0, energyMap.shape[1]):
                energyMap[i][j] += np.min(energyMap[i-1, max(j-1,0):j+2])
    else:
        raise Exception("Argument seamDirection must be either 'HORIZONTAL' or 'VERTICAL'. Provided:\n%s" % seamDirection)
    return energyMap

