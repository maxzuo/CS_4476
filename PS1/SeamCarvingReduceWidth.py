import numpy as np
import matplotlib.image as Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cumulativeEnergyMap
import energyImage
import verticalSeam
import horizontalSeam
import displaySeam
import reduceWidth
import reduceHeight

images = [("inputSeamCarvingPrague.jpg", "outputReduceWidthPrague.png"), ("inputSeamCarvingMall.jpg", "outputReduceWidthMall.png")]

def seamCarvingReduceWidth(img, amount=100):
    orig_e = e = energyImage.energy_image(img)

    for _ in range(amount):
        img, e = reduceWidth.reduceWidth(img, e)

    return orig_e, img


if __name__ == "__main__":
    for src, outpath in images:
        print('src:', src)
        orig_img = img = Image.imread(src)
        e, img = seamCarvingReduceWidth(img, amount=100)
        plt.imsave(outpath, img)
        plt.imsave("energy_"+outpath, e)

        # Cumulative map
        c_v = cumulativeEnergyMap.cumulative_minimum_energy_map(e, "vertical")

        # First seams
        vs = verticalSeam.find_optimal_vertical_seam(c_v)

        plt.imsave("c_v_"+outpath, c_v)

        vs_fig = displaySeam.displaySeam(orig_img, vs, 'vertical')

        vs_fig.savefig("vs_fig_"+outpath)
    
    plt.show()