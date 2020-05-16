import numpy as np
import cumulativeEnergyMap
import horizontalSeam

def reduceHeight(im, energyImage):
    c_h = cumulativeEnergyMap.cumulative_minimum_energy_map(energyImage, "vertical")
    hs = horizontalSeam.find_optimal_horizontal_seam(c_h)
    
    
    im = np.rot90(im)
    
    im = np.array([np.delete(row, hs[i], axis=0) for i, row in enumerate(im)])
    energyImage = np.array([np.delete(row, hs[i]) for i, row in enumerate(energyImage.T)]).T
    
    im = np.rot90(im.astype(np.uint8), 3)
    
    return im, energyImage