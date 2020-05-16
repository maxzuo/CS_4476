import numpy as np
import matplotlib.image as Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cumulativeEnergyMap
import energyImage
import verticalSeam
import horizontalSeam
import displaySeam
import reduceHeight

images = [("inputSeamCarvingPrague.jpg", "outputReduceHeightPrague.png"), ("inputSeamCarvingMall.jpg", "outputReduceHeightMall.png")]

def seamCarvingReduceHeight(img, amount=100):
    orig_e = e = energyImage.energy_image(img)

    for _ in range(amount):
        img, e = reduceHeight.reduceHeight(img, e)

    return orig_e, img


if __name__ == "__main__":
    for src, outpath in images:
        print('src:', src)
        orig_img = img = Image.imread(src)
        e, img = seamCarvingReduceHeight(img, amount=100)
        plt.imsave(outpath, img)
        plt.imsave("energy_"+outpath, e)

        # Cumulative map
        c_h = cumulativeEnergyMap.cumulative_minimum_energy_map(e, "horizontal")

        # First seams
        hs = horizontalSeam.find_optimal_horizontal_seam(c_h)
        
        plt.imsave("c_h_"+outpath, c_h)

        hs_fig = displaySeam.displaySeam(orig_img, hs, 'horizontal')

        hs_fig.savefig("hs_fig_"+outpath)
    
    plt.show()