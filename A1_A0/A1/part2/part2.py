#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import sys
import numpy as np
from PIL import Image


add = sys.argv[1]
img = Image.open(add)
img = np.array(img)

def save(array, name):
    k = Image.fromarray(array.astype(np.uint8))
    k.save(f'{name}.jpg')

def show(array):
    array = np.array(array)/np.max(array)*255
    data = Image.fromarray(array.astype(np.uint8))
    data.show()

def lenCalc(medianListR, medianListG, medianListB):
    dmax = 0
    for i in range(len(medianListR)-1):
        d = medianListR[i+1] - medianListR[i]
        if d>dmax:
            dmax = d
            ini = medianListR[i]
            last = medianListR[i+1]
            channel = 0
            

    for i in range(len(medianListG)-1):
        d = medianListG[i+1] - medianListG[i]
        if d>dmax:
            dmax = d
            ini = medianListG[i]
            last = medianListG[i+1]
            channel = 1
            
            
    for i in range(len(medianListB)-1):
        d = medianListB[i+1] - medianListB[i]
        if d>dmax:
            dmax = d
            ini = medianListB[i]
            last = medianListB[i+1]
            channel = 2
            
    return int(ini), int(last), int(channel)### ini and last is color value

def avgMapping(medianListR, medianListG, medianListB):
    avgR = []
    avgG = []
    avgB = []
    
    for i in range(len(medianListR)-1):
        avg = (medianListR[i+1] + medianListR[i])/2
        avgR.append(avg)

    for i in range(len(medianListG)-1):
        avg = (medianListG[i+1] + medianListG[i])/2
        avgG.append(avg)
            
    for i in range(len(medianListB)-1):
        avg = (medianListB[i+1] + medianListB[i])/2
        avgB.append(avg)
        
    return avgR, avgG, avgB

def medianCut(img, passes=4):
    R = img[:,:,0]
    R = R.flatten()
    
    G = img[:,:,1]
    G = G.flatten()
    
    B = img[:,:,2]
    B = B.flatten()
    
    R.sort()
    G.sort()
    B.sort()
    
    medR = []
    medG = []
    medB = []
    
    medR.append(np.min(img[:,:,0]))
    medR.append(np.max(img[:,:,0]))
    
    medG.append(np.min(img[:,:,1]))
    medG.append(np.max(img[:,:,1]))

    medB.append(np.min(img[:,:,2]))
    medB.append(np.max(img[:,:,2]))

    
    for i in range(passes):
        
        ini, last, channel = lenCalc(medR, medG, medB)
        if channel == 0:
            for n in range(len(R)):
                if ini == R[n]:
                    st = n
                if last == R[n]:
                    en = n
            medR.append((np.median(R[st:en])))### this should be index
            medR.sort()
            
        if channel == 1:
            for n in range(len(G)):
                if ini == G[n]:
                    st = n
                if last == G[n]:
                    en = n
                    
            medG.append((np.median(G[st:en])))### this should be index
            medG.sort() 

        if channel == 2:
            for n in range(len(B)):
                if ini == B[n]:
                    st = n
                if last == B[n]:
                    en = n
            medB.append((np.median(B[st:en])))### this should be index
            medB.sort()
        
        avgR, avgG, avgB = avgMapping(medR, medG, medB)
        
    medR = np.array(medR).astype(np.float64)
    medG = np.array(medG).astype(np.float64)
    medB = np.array(medB).astype(np.float64)

    x = np.shape(img)[0]
    y = np.shape(img)[1]
    z = np.shape(img)[2]
    for row in range(x):
        for column in range(y):
            for channel in range(z):
                if channel == 0:
                    for m in range(1, len(medR)):
                        if int(img[row, column, channel])<int(medR[m]) and int(img[row, column, channel])>int(medR[m-1]):
                            
                            img[row, column, channel] = avgR[m-1]
                            break
                        
                if channel == 1:
                    for m in range(1,len(medG)) :
                        if img[row, column, channel]<medG[m] and img[row, column, channel]>medG[m-1]:
                            img[row, column, channel] = avgG[m-1]
                            break
                if channel == 2:
                    for m in range(1,len(medB)):
                        if img[row, column, channel]<medB[m] and img[row, column, channel]>medB[m-1]:
                            img[row, column, channel] = avgB[m-1]
                            break
    return img

n = medianCut(img, 4)
show(n)
save(n, f'medianCut image')

###################################~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~################################

def spreadError(error, x, y, k, image, shape = np.shape(img)):
    row, column, channel = shape
    if y + 1 < column:
        image[x, y+1, k] = image[x, y+1, k] + error * 7 / 16
        
    if x + 1 < row:
        if y - 1 >= 0:
            image[x + 1, y - 1, k] = image[x + 1, y - 1, k] + error * 3 / 16
        image[x + 1, y, k] = image[x + 1, y, k] + error * 5 / 16
        if y + 1 < column:
            image[x + 1, y + 1, k] = image[x + 1, y + 1, k] + error * 1 / 16
            
    return image

def floydSteinberg(img, bitMap):
    x, y, z = np.shape(img)
    for i in range(x):
        for j in range(y):
            for k in range(z):
                quantError = img[i, j, k] - bitMap[i, j, k]
                img = spreadError(quantError.astype(np.uint8), i, j, k, img)
    return img

# add = sys.argv[1]
img = Image.open(add)
img = np.array(img)
m = floydSteinberg(img, n)
show(m)
save(m, f'floyd-dithering image')

