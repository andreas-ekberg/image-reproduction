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
    for i in range(50):
        data_image = first_data[i].reshape((3, 32, 32)).transpose(1, 2, 0)
        data_images.append(data_image)
    return data_images


def calculateColorAverage(pictureSample):
    return np.mean(pictureSample, axis=(0, 1))

images = loadData200()

def export_data_images(data_images, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data_images, file)




index_table=[]
for pic in images:
    avgRGB = calculateColorAverage(pic)
    avgLAB = color.rgb2lab(avgRGB)
    index_table.append(avgLAB)

print(len(index_table))
np.savetxt('indexArray.csv', index_table, delimiter=',')

export_data_images(images, "data_images.pkl")