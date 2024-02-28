import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io, color
import math 

def calculateColorAverage(pictureSample):
    return np.mean(pictureSample,axis=(0,1))

def euclidianLabDif(ogImg, refImg):
    dist = np.sqrt(np.sum((ogImg - refImg)**2))
    return dist


def load_data_images(filename):
    with open(filename, 'rb') as file:
        data_images = pickle.load(file)
    return data_images

def main():
    #Load different data: 
    #   images is the 200 small images
    #   index_table is the table with the average LAB of the 200 small images
    #   gnuImg is the original image   
    images = load_data_images("data_images.pkl")
    index_table = np.loadtxt('indexArray.csv', delimiter=',')

    #Load the image and get height and width
    gnuImg = io.imread("gnu2.jpg")
    gnuAvg = calculateColorAverage(gnuImg)
    #gnuLAB = color.rgb2lab(gnuAvg)

    #Get the size of the loaded image
    height, width, channels = gnuImg.shape
    #Calculate the amout for padding and then pad the image
    Xpadding = width % 32
    Ypadding = height % 32
    paddedImage  = np.pad(gnuImg, ((Ypadding,Ypadding), (Xpadding,Xpadding),(0,0)), "constant", constant_values=0)
    padHeight, padWidth, channels = paddedImage.shape

    #Final image with all the small images
    addedPhotosArray = np.zeros((padHeight, padWidth, 3), dtype=int)
    finalIndex = 0


    #Stores all the indexes of the images that are close in LAB
    smallest_diff_index =[]

    for partY in range(round(padHeight/32)):
        skipY = partY*32
        for partX in range(round(padWidth/32)):
            skipX = partX*32
            smallest_diff = math.inf

            #This was the double for loop to take a 32x32 section of the original image
            imageSection = paddedImage[skipY:skipY+32, skipX:skipX+32]
        
            for i in range(len(index_table)): 
                originalImageLab = color.rgb2lab(calculateColorAverage(imageSection))
                dist = euclidianLabDif(originalImageLab, index_table[i])
                if(dist < smallest_diff):
                    index = i
                    smallest_diff = dist

            smallest_diff_index.append(index)

            addedPhotosArray[skipY:skipY+32, skipX:skipX+32] = images[smallest_diff_index[finalIndex]]
            finalIndex += 1
        progress = (partY/ round(padHeight/32)) * 100
        print(f"Progress: {progress:.2f}% done")

    plt.imshow(addedPhotosArray)
    #plt.imshow(gnuImg)
    plt.axis('off')  # Hide axes
    plt.show()

    # Now, loaded_images is a NumPy array containing the original images



main()