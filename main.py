import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io, color
import math 

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

def load_data_images(filename):
    with open(filename, 'rb') as file:
        data_images = pickle.load(file)
    return data_images

""" def addPhotos():
    for partY in range(2):
        skipY = partY*32
        for partX in range(2):
            skipX = partX*32
            for i in range(32):
                for j in range(32):
                    addedPhotosArray[i+skipY,j+skipX] = images[smallest_diff_index[finalIndex]][i-skipY,j-skipX]
            finalIndex += 1 """

def main():
    sizeX = 1024
    sizeY = 1024

    images = load_data_images("data_images.pkl")
    index_table = np.loadtxt('indexArray.csv', delimiter=',')
    gnuImg = io.imread("gnu2.jpg")

    addedPhotosArray = np.zeros((sizeX, sizeY, 3), dtype=int)
    finalIndex = 0

    output_images_array = []

    smallest_diff = math.inf
    smallest_diff_index =[]

    for partY in range(32):
        skipY = partY*32
        for partX in range(32):
            skipX = partX*32


            testBit = np.zeros((32, 32, 3))
            for i in range(32):
                for j in range(32):
                    testBit[i,j] = gnuImg[i+skipY,skipX+j]
                    #output_img[i,skip+j]= testBit
        
            output_images_array.append(testBit)

            for i in range(len(index_table)): 
                dist = euclidianLabDif(color.rgb2lab(calculateColorAverage(testBit)), index_table[i])
                if(dist < smallest_diff):
                    #print("found")
                    index = i
                    smallest_diff = dist

            smallest_diff_index.append(index)

            for k in range(32):
                for m in range(32):
                    addedPhotosArray[k+skipY,m+skipX] = images[smallest_diff_index[finalIndex]][k,m]
            finalIndex += 1


    
    
    
    """ for output_index in range(len(output_images_array)):
        for i in range(len(index_table)): 
            dist = euclidianLabDif(color.rgb2lab(calculateColorAverage(output_images_array[output_index])), index_table[i])
            if(dist < smallest_diff):
                #print("found")
                index = i
                smallest_diff = dist
        smallest_diff_index.append(index) """




    #print(smallest_diff_index)
    
    """ for partY in range(4):
        skipY = partY*32
        for partX in range(4):
            skipX = partX*32
            for i in range(32):
                for j in range(32):
                    addedPhotosArray[i+skipY,j+skipX] = images[smallest_diff_index[finalIndex]][i-skipY,j-skipX]
            finalIndex += 1 """

    #print(smallest_diff_index)
    #height, width, channels = test.shape
    #plt.text(10, 10, f"Dimensions: {width}x{height}\nChannels: {channels}", color='white', fontsize=8, ha='left', va='top', bbox=dict(facecolor='black', alpha=0.7))
    #print(lab)

    plt.imshow(addedPhotosArray)
    #plt.imshow(gnuImg)
    plt.axis('off')  # Hide axes
    plt.show()

    # Now, loaded_images is a NumPy array containing the original images



main()