import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io, color
import math
from sklearn.cluster import KMeans

def getColorPalette():
    image = mpimg.imread('wilma3.jpg')
    w, h, d = tuple(image.shape)
    pixel = np.reshape(image, (w * h, d))
    n_colors = 10
    
    model = KMeans(n_clusters=n_colors, random_state=42).fit(pixel)
    
    # Get the cluster centers
    colour_palette = model.cluster_centers_

    return colour_palette


def load_cifar10_batch(file_path):
    with open(file_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    return batch


def rgb2int(r, g, b):
    colorInt = 256 * 256 * r + 256 * g + b
    return colorInt


def load_cifar10_data():
    data = []
    labels = []

    for batch_num in range(1, 6):
        file_path = f"cifar-10-batches-py/data_batch_{batch_num}"
        batch = load_cifar10_batch(file_path)
        data.append(batch[b"data"])
        labels.extend(batch[b"labels"])

    test_batch = load_cifar10_batch("cifar-10-batches-py/test_batch")
    test_data = test_batch[b"data"]
    test_labels = test_batch[b"labels"]

    data = np.concatenate(data)
    labels = np.array(labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    return data, labels, test_data, test_labels


# Load CIFAR-10 data
first_data, labels, test_data, test_labels = load_cifar10_data()

def calculateColorAverage(pictureSample):
    return np.mean(pictureSample, axis=(0, 1))


def euclidianLabDif(ogImg, refImg):
    dist = np.sqrt(np.sum((ogImg - refImg) ** 2))
    return dist

def loadData2(refAvgLAB):
    data_images = []
    amountOfPictures = 20
    wentThrough = 0
    amountSkip = 1
    #Goes through the color palette
    for labPalette in refAvgLAB:
        #For each cluster center in color palette, go through all the images in CIFAR
        for i in range(len(first_data)):
            data_image = first_data[i].reshape((3, 32, 32)).transpose(1, 2, 0)
            dist = abs(
                euclidianLabDif(
                    labPalette, color.rgb2lab(calculateColorAverage(data_image))
                )
            )
            if dist < 950:#The threshold
                data_images.append(data_image)#Add the image

            if len(data_images) == amountOfPictures*amountSkip:
                print(amountOfPictures, " images are added")
                break
            wentThrough += 1
        print("Went through ", wentThrough, "amount of pictures")
        wentThrough = 0
        print("Data Images has size: ", len(data_images))
        amountSkip += 1
    return data_images

color_palette = getColorPalette()
ny_palette = []
for i in range(len(color_palette)):
    ny_palette.append(color.rgb2lab(color_palette[i]))

images = loadData2(ny_palette)



def export_data_images(data_images, filename):
    with open(filename, "wb") as file:
        pickle.dump(data_images, file)


index_table = []
for pic in images:
    avgRGB = calculateColorAverage(pic)
    avgLAB = color.rgb2lab(avgRGB)
    index_table.append(avgLAB)

print("Length of index table", len(index_table))

np.savetxt("indexArray.csv", index_table, delimiter=",")

export_data_images(images, "data_images.pkl") 