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

def euclidianLabDif(ogImg, refImg):
    dist = 0; 
    L_diff = pow(ogImg[0] - refImg[0],2)
    A_diff = pow(ogImg[1] - refImg[1],2)
    B_diff = pow(ogImg[2]- refImg[2],2)
    dist = math.sqrt(L_diff+A_diff+B_diff)
    return dist


def main():
    images = loadData200()
    
    gnuImg = io.imread("gnu2.jpg")
    gnuLab = color.rgb2lab(gnuImg)
    output_img = np.zeros((32, 64, 3))
    output_images_array = []
    for part in range(2):
        skip = part*32
        testBit = np.zeros((32, 32, 3))
        for i in range(32):
            for j in range(32):
                testBit[i,j] = gnuLab[i+2290,1950+j]
                #output_img[i,skip+j]= testBit
        output_images_array.append(testBit)


    help = calculateColorAverage(output_images_array[0])
    lab = color.rgb2lab(help)

    index_table = np.loadtxt('indexArray.csv', delimiter=',')
    smallest_diff = math.inf
    smallest_diff_index =[]
    addedPhotosArray = np.zeros((32, 64, 3), dtype=int)
    for output_index in range(len(output_images_array)):
        for i in range(len(index_table)): 
            dist = euclidianLabDif(calculateColorAverage(output_images_array[output_index]), index_table[i])
            if(dist < smallest_diff):
                #print("found")
                smallest_diff = dist
                smallest_diff_index.append(i)

    for i in range(32):
        for j in range(64):
            if(j <32):
                addedPhotosArray[i,j] = images[smallest_diff_index[1]][i,j]
            else:
                addedPhotosArray[i,j] = images[smallest_diff_index[1]][i,j-32]

    print(smallest_diff_index)
    #height, width, channels = test.shape
    #plt.text(10, 10, f"Dimensions: {width}x{height}\nChannels: {channels}", color='white', fontsize=8, ha='left', va='top', bbox=dict(facecolor='black', alpha=0.7))
    #print(lab)

    plt.imshow(addedPhotosArray)
    plt.axis('off')  # Hide axes
    plt.show()

    # Now, loaded_images is a NumPy array containing the original images



main()