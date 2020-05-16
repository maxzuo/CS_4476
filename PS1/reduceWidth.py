import numpy as np
import cumulativeEnergyMap
import verticalSeam

def reduceWidth(im, energyImage):
    c_v = cumulativeEnergyMap.cumulative_minimum_energy_map(energyImage, "vertical")
    vs = verticalSeam.find_optimal_vertical_seam(c_v)
    
    im = np.array([np.delete(row, vs[i], axis=0) for i, row in enumerate(im)])
    energyImage = np.array([np.delete(row, vs[i]) for i, row in enumerate(energyImage)])
    
    return im.astype(np.uint8), energyImage
