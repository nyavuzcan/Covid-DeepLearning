
from PIL import Image
import os
import numpy as np

path='data/datasetall'
fourFile = os.listdir('data/datasetall')

#FOR GRAY SCALE
allDataSet = []
rgbValues = [0.2989, 0.5870, 0.1140]
arr = np.array([[1, 2, 3], [4, 5, 6]], np.int32)

#IMAGE ROTATE

for oneFolder in fourFile:
    if (oneFolder == 'covid'):

            covidFiles = os.listdir(path+'/'+oneFolder+'/')
            i=0
            for oneCovidFile in covidFiles:

                onePath=path+'/'+oneFolder+'/'+oneCovidFile

                if onePath.endswith(".jpeg") or onePath.endswith(".png") or onePath.endswith(".jpg"):
                    print(onePath)
                    imageForRotate = Image.open(onePath)
                    degree = 45
                    #90,180,270,45,135,225
                    outPut = imageForRotate.rotate(degree)

                    outPut.save(path+'/'+oneFolder+'/'+str(i)+str(degree)+'.png')
                    print(outPut)
                    i+=1

