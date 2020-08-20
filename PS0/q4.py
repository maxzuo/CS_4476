import numpy as np
import matplotlib.pyplot as plt
import imageio as Image
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
    
    image = Image.imread("q4-input.png")

    # (a)
    swapped = swap_red_green(image)
    plt.imsave("q4-output-swapped.png", swapped)

    # (b)
    grayscale = rgb2gray(image)
    plt.imsave("q4-output-grayscale.png", grayscale, cmap=cm.gray)

    # (c)
    negative = negative(grayscale)
    plt.imsave("q4-output-negative.png", negative, cmap=cm.gray)

    # (d)
    mirrored = mirror_horizontally(grayscale)
    plt.imsave("q4-output-mirrored.png", mirrored, cmap=cm.gray)

    # (e)
    averaged = average(grayscale, mirrored)
    plt.imsave("q4-output-average.png", averaged, cmap=cm.gray)

    # (f)
    N = np.random.randint(0, 255, grayscale.shape)
    np.save("q4-noise.npy", N)

    added = add(grayscale, N)
    plt.imsave("q4-output-noise.png", added, cmap=cm.gray)


    # Final plot
    plt.axis("off")
    fig, axes = plt.subplots(3, 2, figsize=(6, 10))
    imgs = [(swapped, "q4-output-swapped.png (a)", None), (grayscale, "q4-output-grayscale.png (b)", cm.gray), (negative, "q4-output-negative.png (c)", cm.gray),
            (mirrored, "q4-output-mirrored.png (d)", cm.gray), (averaged, "q4-output-average.png (e)", cm.gray), (added, "q4-otuput-noise.png (f)", cm.gray)]

    axes = np.array(axes).flatten().tolist()

    for (img, title, cmap), ax in zip(imgs, axes):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)

        ax.axis("off")

    fig.savefig("all_results.png")
    plt.show()


