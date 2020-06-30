
from PIL import Image
import os
from keras.preprocessing.image import  img_to_array, load_img
import numpy as np

path='data/datasetall'
fourFile = os.listdir('data/datasetall')

#FOR GRAY SCALE
allDataSet = []
rgbValues = [0.2989, 0.5870, 0.1140]
arr = np.array([[1, 2, 3], [4, 5, 6]], np.int32)

#Covid is one class and the others are one class
for oneFolder in fourFile:
    if (oneFolder == 'bacterianormalvirus'):
        bacteriaFiles = os.listdir(path+'/'+oneFolder+'/')
        for oneBacteriFile in bacteriaFiles:
            onePath=path+'/'+oneFolder+'/'+oneBacteriFile

            if onePath.endswith(".jpeg") or onePath.endswith(".png"):
                print(onePath)
                image = load_img(onePath,target_size=(28,28,3))
                arr= img_to_array(image)
                arr = np.dot(arr,rgbValues)

                imageNewArr = Image.fromarray(arr)
                allDataSet.append([arr,0])

    if (oneFolder == 'covid'):
        covidFiles = os.listdir(path+'/'+oneFolder+'/')
        for oneCovidFile in covidFiles:
            onePath=path+'/'+oneFolder+'/'+oneCovidFile

            if onePath.endswith(".jpeg") or onePath.endswith(".png"):
                print(onePath)
                image = load_img(onePath,target_size=(28,28,3))
                arr= img_to_array(image)
                arr = np.dot(arr,rgbValues)
                imageNewArr = Image.fromarray(arr)
                allDataSet.append([arr,1])

datasetForSave = np.array(allDataSet)

np.save("datasetTwoClasses.npy",datasetForSave)

