
from PIL import Image
import os
from keras.preprocessing.image import img_to_array, load_img
import numpy as np


path='data/datasetall'
fourFile = os.listdir('data/datasetall')

#FOR GRAY SCALE
allDataSet = []
rgbValues = [0.2989, 0.5870, 0.1140]
arr = np.array([[1, 2, 3], [4, 5, 6]], np.int32)


for oneFolder in fourFile:
    if (oneFolder == 'bacteria'):
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

    if (oneFolder == 'normal'):
        normalFiles = os.listdir(path+'/'+oneFolder+'/')
        for oneNormalFile in normalFiles:

            onePath=path+'/'+oneFolder+'/'+oneNormalFile
            if onePath.endswith(".jpeg") or onePath.endswith(".png"):
                print(onePath)
                image = load_img(onePath,target_size=(28,28,3))
                arr= img_to_array(image)
                arr = np.dot(arr,rgbValues)

                imageNewArr = Image.fromarray(arr)
                allDataSet.append([arr,2])

    if (oneFolder == 'virus'):
        virusFiles = os.listdir(path+'/'+oneFolder+'/')
        for oneVirusFile in virusFiles:
            onePath=path+'/'+oneFolder+'/'+oneVirusFile

            if onePath.endswith(".jpeg") or onePath.endswith(".png"):
                print(onePath)
                image = load_img(onePath,target_size=(28,28,3))
                arr= img_to_array(image)
                arr = np.dot(arr,rgbValues)

                imageNewArr = Image.fromarray(arr)
                allDataSet.append([arr,3])

datasetForSave = np.array(allDataSet)

np.save("datasetFourClasses.npy",datasetForSave)

