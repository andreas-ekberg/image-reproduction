import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io, color
import math


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

first_data, labels, test_data, test_labels = load_cifar10_data()


def calculateColorAverage(pictureSample):
    return np.mean(pictureSample, axis=(0, 1))


def euclidianLabDif(ogImg, refImg):
    dist = np.sqrt(np.sum((ogImg - refImg) ** 2))
    return dist


def loadData():
    data_images = []
    amountOfPictures = 50
    wentThrough = 0

    #Goes through all the pictures loaded from the CIFAR
    for i in range(len(first_data)):
        data_image = first_data[i].reshape((3, 32, 32)).transpose(1, 2, 0)
        if len(data_images) == 0:
            data_images.append(data_image)
        else:
            avgRGB = calculateColorAverage(data_image)
            avgLAB = color.rgb2lab(avgRGB)
            counter = 1
            #Goes through the array where the chosen images has been added.
            for addedImg in data_images:
                dist = abs(
                    euclidianLabDif(
                        avgLAB, color.rgb2lab(calculateColorAverage(addedImg))
                    )
                )
                if dist < 1700: # The threshold
                    break
                elif len(data_images) == counter:
                    data_images.append(data_image) #Adds to the array of chosen images.
                counter += 1

            if len(data_images) == amountOfPictures:
                print(amountOfPictures, " images are added")
                break
        wentThrough += 1
    print("Went through ", wentThrough, "amount of pictures")
    return data_images


images = loadData()


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
