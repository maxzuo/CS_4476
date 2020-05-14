import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
import matplotlib.cm as cm

image = None

# def swap_red_green(image):
#     swapped = np.array(image)

#     swapped[:, :, 0], swapped[:, :, 1] = swapped[:, :, 1], swapped[:, :, 0]

#     return swapped

def swap_red_green(image):
    return image @ np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.uint8)

def rgb2gray(image):
    #return np.mean(rgb, -1)
    return np.dot(image[...,:], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

def negative(image):
    return 255 - image

def mirror_horizontally(image):
    return np.array(image[:, ::-1,...])

def average(img, img2):
    return ((img.astype(np.double) + img2.astype(np.double)) // 2).astype(np.uint8)

def add(img, img2):
    return ((img.astype(np.double) + img2.astype(np.double))).clip(255).astype(np.uint8)

if __name__ == "__main__":
    
    image = Image.imread("inputPS0Q2.jpg")

    # 1
    swapped = swap_red_green(image)
    plt.imsave("swapImgPS0Q2.png", swapped)

    # 2
    grayscale = rgb2gray(image)
    plt.imsave("grayImgPS0Q2.png", grayscale, cmap=cm.gray)

    # 3a
    negative = negative(grayscale)
    plt.imsave("negativeImgPS0Q2.png", negative, cmap=cm.gray)

    # 3b
    mirrored = mirror_horizontally(grayscale)
    plt.imsave("mirrorImgPS0Q2.png", mirrored, cmap=cm.gray)

    # 3c
    averaged = average(grayscale, mirrored)
    plt.imsave("avgImgPS0Q2.png", averaged, cmap=cm.gray)

    # 3d
    N = np.random.randint(0, 255, grayscale.shape)
    np.save("noise.npy", N)

    added = add(grayscale, N)
    plt.imsave("addNoiseImgPS0Q2.png", added, cmap=cm.gray)


    # Final plot
    plt.axis("off")
    fig, axes = plt.subplots(3, 2, figsize=(6, 10))
    imgs = [(swapped, "swapImgPS0Q2.png (1)", None), (grayscale, "grayImgPS0Q2.png (2)", cm.gray), (negative, "negativeImgPS0Q2.png (3a)", cm.gray),
            (mirrored, "mirrorImgPS0Q2.png (3b)", cm.gray), (averaged, "avgImgPS0Q2.png (3c)", cm.gray), (added, "addNoiseImgPS0Q2.png (3d)s", cm.gray)]

    axes = np.array(axes).flatten().tolist()

    for (img, title, cmap), ax in zip(imgs, axes):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)

        ax.axis("off")

    fig.savefig("all_results.png")
    plt.show()


