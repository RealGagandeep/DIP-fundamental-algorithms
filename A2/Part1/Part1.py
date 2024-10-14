#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from PIL import Image, ImageFilter
import numpy as np
import cv2

def save(array, name):
    k = Image.fromarray(array.astype(np.uint8))
    k.save(f'{name}.jpg')

def show(array):
#     array = np.array(array)/np.max(array)*255
    data = Image.fromarray(array.astype(np.uint8))
    data.show()

def c(x,u):
    if u == 0:
        k = 1/np.sqrt(8)
    else:
        k = 1/2
    return k*np.cos((2*x+1)*u*np.pi/16)

def cDash(x,u):
    return (-u*np.pi/8)*np.sin((2*x+1)*u*np.pi/16)

def convolve(img):
    resx = 0
    resy = 0
    valx = []
    valy = []
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            resx = resx + cDash(xpas, i)*c(ypas, j)*img[i, j]
            resy = resy + c(xpas, i)*cDash(ypas, j)*img[i, j]
    valx.append(resx)
    valy.append(resy)
    valx = np.reshape(valx,np.shape(img))
    valy = np.reshape(valy,np.shape(img))
    return valx, valy
                    
def infoMap(img):
    
    newImg = np.zeros((xLen, yLen))
    newImg[:np.shape(img)[0], :np.shape(img)[1]] = img[:,:]
    image = np.zeros((xLen, yLen))
    for i in range(0,np.shape(newImg)[0]):
        for j in range(0,np.shape(newImg)[1]):
            imageX[i:i,j:j], imageY[i:i,j:j] = convolve(img[i:i, j:j])
            
    image = image.T
    for i in range(0,np.shape(newImg)[0]):
        for j in range(0,np.shape(newImg)[1]):
            imageX[i:i,j:j], imageY[i:i,j:j] = convolve(img[i:i, j:j])
        print(i)
    return image


def convolution(fltr, image, row, column, FilterSize):
    return np.sum(fltr*image[row:row+FilterSize, column:column+FilterSize])


def small(img, i, j):
    if j == 0:
        if img[i,j]>=img[i,j+1]:
            return img[i,j+1]
        else:
            return img[i,j]
        
    if j >= np.shape(img)[1]-1:
        if img[i,j]>=img[i,j-1]:
            return img[i,j-1]
        else:
            return img[i,j]
        
    else:
        if img[i,j]>=img[i,j+1] and img[i,j-1]>=img[i,j+1]:
            return img[i,j+1]
        
        elif img[i,j+1]>=img[i,j] and img[i,j-1]>=img[i,j]:
            return img[i,j]
        
        elif img[i,j]>=img[i,j-1] and img[i,j+1]>=img[i,j-1]:
            return img[i,j-1]

def energyMap(img):
    img = np.array(img)
    x = np.shape(img)[0]
    y = np.shape(img)[1]
    for i in range(x-1,0,-1):
        for j in range(y):
            img[i-1,j] = img[i-1,j] + small(img, i, j)
    return img

def path(img):
    path = []
    cordinates = []
    minval = 10**7
    for m in range(np.shape(img)[1]):
        if img[0,m]<=minval:
#             print(m)
            minval = img[0,m]
            ini = m
    path.append(ini)
    cordinates.append((0,ini))
    for i in range(1,np.shape(img)[0]):
        if ini <= 0:
            if img[i,ini]<=img[i,ini+1]:
                path.append(ini)
                cordinates.append((i,ini))
            else:
                ini+=1
                path.append(ini)
                cordinates.append((i,ini))
            
            
        elif ini >= np.shape(img)[1]-1:
            if img[i,ini]<=img[i,ini-1]:
                path.append(ini)
                cordinates.append((i,ini))
            else:
                ini-=1
                path.append(ini)
                cordinates.append((i,ini))
        elif ini > 0 and ini < np.shape(img)[1]-1:
            if img[i,ini]<=img[i,ini+1] and img[i,ini-1]>=img[i,ini]:
                path.append(ini)
                cordinates.append((i,ini))
            elif img[i,ini]>=img[i,ini+1] and img[i,ini-1]>=img[i,ini+1]:
                ini += 1
                path.append(ini)
                cordinates.append((i,ini))
            elif img[i,ini]>=img[i,ini-1] and img[i,ini-1]<=img[i,ini+1]:
                ini -= 1
                path.append(ini)
                cordinates.append((i,ini))
    return path, cordinates

def marker(img, points):
    img = img.astype(np.int32)
    for i in range(len(points)):
        (x, y) = points[i]
        img[x, y,0] = -1
        img[x, y,1] = 0
        img[x, y,2] = 0
#         img[x, y,1] = 255
#         img[x, y,2] = -1
    
    return img


def marker1(img, points):
    img = img.astype(np.int32)
    for i in range(len(points)):
        (x, y) = points[i]
        img[x, y] = -1
#         img[x, y,1] = 0
#         img[x, y,2] = 0
    
    return img

def remove(img):
    newImg = []
#     print(np.shape(img))
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            if img[i,j,0] != -1:
                newImg.append(img[i,j])
    newImg = np.reshape(np.array(newImg),(np.shape(img)[0],np.shape(img)[1]-1,3))
    return newImg

def remove1(img):
    newImg = []
#     print(np.shape(img))
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            if img[i,j] != -1:
                newImg.append(img[i,j])
    newImg = np.reshape(np.array(newImg),(np.shape(img)[0],np.shape(img)[1]-1))
    return newImg



def smallH(img, i, j):
    if i == 0:
        if img[i,j]>=img[i+1,j]:
            return img[i+1,j]
        else:
            return img[i,j]
        
    if i >= np.shape(img)[0]-1:
        if img[i,j]>=img[i-1,j]:
            return img[i-1,j]
        else:
            return img[i,j]
        
    else:
        if img[i,j]>=img[i+1,j] and img[i-1,j]>=img[i+1,j]:
            return img[i+1,j]
        
        elif img[i+1,j]>=img[i,j] and img[i-1,j]>=img[i,j]:
            return img[i,j]
        
        elif img[i,j]>=img[i-1,j] and img[i+1,j]>=img[i-1,j]:
            return img[i-1,j]

def energyMapH(img):
    img = np.array(img)
    x = np.shape(img)[0]
    y = np.shape(img)[1]
    for j in range(y-1,0,-1):
        for i in range(x):
            img[i,j-1] = img[i,j-1] + smallH(img, i, j)
    return img

def pathH(img):
    path = []
    cordinates = []
    minval = 10**7
    for m in range(np.shape(img)[0]):
        if img[m,0]<=minval:
#             print(m)
            minval = img[m,0]
            ini = m
    path.append(ini)
    cordinates.append((ini,0))
    for i in range(1,np.shape(img)[1]):
        if ini <= 0:
            if img[ini,i]<=img[ini+1,i]:
                path.append(ini)
                cordinates.append((ini,i))
            else:
                ini+=1
                path.append(ini)
                cordinates.append((ini,i))
            
            
        elif ini >= np.shape(img)[0]-1:
            if img[i,ini]<=img[ini-1,i]:
                path.append(ini)
                cordinates.append((ini,i))
            else:
                ini-=1
                path.append(ini)
                cordinates.append((ini,i))
        elif ini > 0 and ini < np.shape(img)[0]-1:
            if img[ini,i]<=img[ini+1,i] and img[ini-1,i]>=img[ini,i]:
                path.append(ini)
                cordinates.append((ini,i))
            elif img[ini,i]>=img[ini+1,i] and img[ini-1,i]>=img[ini+1,i]:
                ini += 1
                path.append(ini)
                cordinates.append((ini,i))
            elif img[ini,i]>=img[ini-1,i] and img[ini-1,i]<=img[ini+1,i]:
                ini -= 1
                path.append(ini)
                cordinates.append((ini,i))
    return path, cordinates

def markerH(img, points):
    img = img.astype(np.int32)
    for i in range(len(points)):
        (x, y) = points[i]
        img[x, y,0] = -1
        img[x, y,1] = 0
        img[x, y,2] = 0
#         img[x, y,1] = 255
#         img[x, y,2] = -1
    
    return img


def marker1H(img, points):
    img = img.astype(np.int32)
    for i in range(len(points)):
        (x, y) = points[i]
        img[x, y] = -1
#         img[x, y,1] = 0
#         img[x, y,2] = 0
    
    return img

def removeH(img):
    newImg = []
#     print(np.shape(img))
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            if img[i,j,0] != -1:
                newImg.append(img[i,j])
    newImg = np.reshape(np.array(newImg),(np.shape(img)[0]-1,np.shape(img)[1],3))
    return newImg

def remove1H(img):
    newImg = []
#     print(np.shape(img))
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
#             case = (img[i,j] != -1)
#             if case ==False:
#                 print(i)
            if img[i,j] != -1:
                newImg.append(img[i,j])
    newImg = np.reshape(np.array(newImg),(np.shape(img)[0]-1,np.shape(img)[1]))
    return newImg

def vCarving(img, a, val):
    chad = img[:,:,:]
    for i in range(a):
        p, points = path(val)
        img = marker(img,points)
        val = marker1(val,points)
        chad = marker(chad,points)
        val = remove1(val)
        img = remove(img)
#         print(i)
    return img, chad, val

def vCarvingH(img, a, val):
    chad = img[:,:,:]
    for i in range(a):
        p, points = pathH(val)
        img = markerH(img,points)
        val = marker1H(val,points)
        chad = markerH(chad,points)
        val = remove1H(val)
        img = removeH(img)
#         print(i)
    return img, chad, val


def hCarving(img, b, val):
#     b = np.shape(img)[1]//15
    img = np.rot90(img, 1, (0, 1))
    img, chad, val = vCarving(img, b, val)
    img = np.rot90(img, 3, (0, 1))
    return img, chad, val


def seamCarving(img):
    a = np.shape(img)[0]//15
    b = np.shape(img)[1]//15
    print(np.shape(img))

    edgeMap = infoMap(img)
    energyMapHorizontal = energyMapH(edgeMap)
    eMap = energyMapHorizontal/np.max(energyMapHorizontal)*255
#     show(eMap)
    img, chad, eMap = hCarving(img,b,energyMapHorizontal)
    eMap = eMap/np.max(eMap)*255
    save(img, "after hCaring")
    save(chad, "markedh")
    save(eMap, "emapH")
    print(np.shape(img))
    
    edgeMap = infoMap(img)
    energyMapVertical = energyMap(edgeMap)
    
    img, chad, eMap = vCarving(img,a,energyMapVertical)
    eMap = eMap/np.max(eMap)*255
    save(img, "after vCaring")
    save(chad, "markedV")
    save(eMap, "emapV")
    
    return img, chad
    


# In[ ]:


import os
from os import listdir
 
# get the path/directory
pth = "C:\\Users\\GAGANDEEP SINGH\\Desktop\\"
p = "C:\\Users\\GAGANDEEP SINGH\\Downloads\\"
for images in os.listdir(pth):
 
    # check if the image ends with png
    if (images.endswith(".jpg")):
        nme = f"{images}"[:-4]
        print(nme)
        name = nme + ".jpg"
        imgPath = pth + name
        img = Image.open(imgPath)
        img = np.array(img)
#         show(img)
        chadNme = p + nme + "_out"
        img, chad = seamCarving(img)
        show(img)
        show(chad)
#         break
        save(img, chadNme)

