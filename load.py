import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io, color
import math 

def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch

def load_cifar10_data():
    data = []
    labels = []

    for batch_num in range(1, 6):
        file_path = f'cifar-10-batches-py/data_batch_{batch_num}'
        batch = load_cifar10_batch(file_path)
        data.append(batch[b'data'])
        labels.extend(batch[b'labels'])

    test_batch = load_cifar10_batch('cifar-10-batches-py/test_batch')
    test_data = test_batch[b'data']
    test_labels = test_batch[b'labels']

    # Convert data to numpy arrays
    data = np.concatenate(data)
    labels = np.array(labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    return data, labels, test_data, test_labels

# Load CIFAR-10 data
first_data, labels, test_data, test_labels = load_cifar10_data()

def loadData200():
    data_images = []
    for i in range(200):
        data_image = first_data[i].reshape((3, 32, 32)).transpose(1, 2, 0)
        data_images.append(data_image)
    return data_images


def calculateColorAverage(pictureSample):
    avgRed = 0
    avgGreen = 0
    avgBlue = 0
    for x in range(0,32):
        for y in range(0,32):
            avgRed+= pictureSample[x][y][0]
            avgGreen+= pictureSample[x][y][1]
            avgBlue+= pictureSample[x][y][2]
    avgRed = avgRed/(32*32)
    avgGreen =avgGreen/(32*32)
    avgBlue = avgBlue/(32*32)

    avgRGB = [avgRed, avgGreen, avgBlue]
    return avgRGB

images = loadData200()
print(images[1])

flattened_images = [image.flatten() for image in images]


index_table=[]
for pic in images:
    avgRGB = calculateColorAverage(pic)
    avgLAB = color.rgb2lab(avgRGB)
    index_table.append(avgLAB)

np.savetxt('indexArray.csv', index_table, delimiter=',')

