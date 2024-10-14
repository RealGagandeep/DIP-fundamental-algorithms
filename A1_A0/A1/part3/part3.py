#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import sys
import random
import numpy as np
from PIL import Image


add1 = sys.argv[1]
img = Image.open(add1)
img = np.array(img)

# img = Image.open('C:\\Users\\GAGANDEEP SINGH\\Desktop\\2.jpg')
# img = np.array(img)

add2 = sys.argv[2]
imgT = Image.open(add2)
imgT = np.array(imgT)

# imgT = Image.open('C:\\Users\\GAGANDEEP SINGH\\Desktop\\4.jpg')
# imgT = np.array(imgT)

def save(array, name):
    k = Image.fromarray(array.astype(np.uint8))
    k.save(f'{name}.jpg')

def show(array):
    array = np.array(array)/np.max(array)*255
    data = Image.fromarray(array.astype(np.uint8))
    data.show()


def rgb2lab(img):
    return np.array(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))

def lab2rgb(img):
    img = cv2.convertScaleAbs(img)
    return np.array(cv2.cvtColor(img, cv2.COLOR_LAB2RGB))

imgT = np.sum(imgT, axis=2)
imgT = (imgT-np.min(imgT))/np.max(imgT)*(np.max(img[:,:,0])-np.min(img[:,:,0]))+np.min(img[:,:,0])

x,y = np.shape(imgT)
tar = np.zeros((x,y,3))
tar[:,:,0] = imgT[:,:]
tar[:,:,1] = imgT[:,:]
tar[:,:,2] = imgT[:,:]

print(np.shape(tar))
print(np.shape(img))

tr = cv2.convertScaleAbs(tar)
tar = rgb2lab(tr)
img = rgb2lab(img)

def randomJitterGen(img, samples):
    lVal = []
    aVal = []
    bVal = []
    for i in range(samples):
        x = random.randint(0,np.shape(img)[0])
        y = random.randint(0,np.shape(img)[1])
        lVal.append(img[x, y, 0])
        aVal.append(img[x, y, 1])
        bVal.append(img[x, y, 2])
    return lVal, aVal, bVal

def L1Norm(pixel1, pixel2):
    return abs(pixel2-pixel1)

def colorImg(img, tar, samples=20):
    x, y, z = np.shape(tar)
    l,a,b = randomJitterGen(img, samples)
    for row in range(x):
            for column in range(y):
                dmin = 10**3
                for item in range(len(l)):
                    if L1Norm(l[item], tar[row, column, 0]) < dmin:
                        dmin = L1Norm(l[item], tar[row, column, 0])
                        tar[row, column, 1] = a[item]
                        tar[row, column, 2] = b[item]
    return tar

colorImg = lab2rgb(colorImg(img, tar))
show(colorImg)
save(colorImg, f'colored image without swatches')


#############################~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~############################


def randomjitter(img, startY, endY, startX, endX, samples):

    lVal = []
    aVal = []
    bVal = []
    for i in range(samples):
        x = random.randint(startX, endX)
        y = random.randint(startY, endY)
        lVal.append(img[x, y, 0])
        aVal.append(img[x, y, 1])
        bVal.append(img[x, y, 2])
    return lVal, aVal, bVal
        
    
def swatchGen(img, imgT):
    
    l,a,b = randomjitter(img, 1015, 1095, 60, 130, 20)
    for row in range(950,1200):
            for column in range(50,210):
                dmin = 10**3
                for item in range(len(l)):
                    if L1Norm(l[item], imgT[column,row, 0]) < dmin:
                        dmin = L1Norm(l[item], imgT[ column,row, 0])
                        imgT[column,row, 1] = a[item]
                        imgT[column,row,  2] = b[item]

#     l,a,b = randomjitter(img, 400, 480, 750, 810, 20)
#     for row in range(550,800):
#             for column in range(850,1060):
#                 dmin = 10**3
#                 for item in range(len(l)):
#                     if L1Norm(l[item], imgT[column,row, 0]) < dmin:
#                         dmin = L1Norm(l[item], imgT[ column,row, 0])
#                         imgT[column,row, 1] = a[item]
#                         imgT[column,row,  2] = b[item]

    l,a,b = randomjitter(img, 350, 460, 215, 280, 20)
    for row in range(1210,1350):
            for column in range(380,460):
                dmin = 10**3
                for item in range(len(l)):
                    if L1Norm(l[item], imgT[column,row, 0]) < dmin:
                        dmin = L1Norm(l[item], imgT[ column,row, 0])
                        imgT[column,row, 1] = a[item]
                        imgT[column,row,  2] = b[item]
                        
    Lval = []
    Aval = []
    Bval = []
    l,a,b = randomJitterGen(imgT[50:210, 950:1050, :], 10)
    Lval.append(l)
    Aval.append(a)
    Bval.append(b)
    
#     l,a,b = randomJitterGen(imgT[850:1060,550:800,:], 5)
#     Lval.append(l)
#     Aval.append(a)
#     Bval.append(b)

    l,a,b = randomJitterGen(imgT[380:460, 1210:1350,:], 10)
    Lval.append(l)
    Aval.append(a)
    Bval.append(b)
    
    Lval = np.reshape(Lval, (-1, 1))
    Aval = np.reshape(Aval, (-1, 1))
    Bval = np.reshape(Bval, (-1, 1))
    show(lab2rgb(imgT))
    row, column, depth = np.shape(imgT)
    for x in range(row):
        for y in range(column):
            dmin = 10**3
            for item in range(len(Lval)):
                if L1Norm(Lval[item], imgT[x, y, 0]) < dmin:
                    dmin = L1Norm(Lval[item], imgT[x, y, 0])
                    imgT[x, y, 1] = Aval[item]
                    imgT[x, y, 2] = Bval[item]
                    
    return imgT

coloredImg = lab2rgb(swatchGen(img, tar))
show(coloredImg)
save(coloredImg, f'colored image using swatches')

