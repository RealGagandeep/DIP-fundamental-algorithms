#!/usr/bin/env python
# coding: utf-8

# In[8]:


# !pip install Image
# !pip install OpenCV
import cv2
import sys
import numpy as np
from PIL import Image

FilterSize = 10
gamma = 1.5
th = 0.005
lamda = 0.5
ro = .05

add = sys.argv[1]
img = Image.open(add)
img = np.array(img)

T = np.array([[1/3,1/3,1/3],[-np.sqrt(6)/6, -np.sqrt(6)/6, np.sqrt(6)/3],[1/np.sqrt(6), -2/np.sqrt(6), 0]])

V1 = np.zeros((np.shape(img)[0], np.shape(img)[1]))
V2 = np.zeros((np.shape(img)[0], np.shape(img)[1]))
I = np.zeros((np.shape(img)[0], np.shape(img)[1]))


def save(array, name):
    k = Image.fromarray(array.astype(np.uint8))
    k.save(f'{name}.jpg')

def show(array):
    array = np.array(array)/np.max(array)*255
    data = Image.fromarray(array.astype(np.uint8))
    data.show()

def calcS(V1, V2):
    return np.sqrt(V1**2 + V2**2)

def calcH(V1, V2):
    l = []
    for x in range(np.shape(V1)[0]):
        for y in range(np.shape(V1)[1]):
            if V2[x, y] == 0:
                V2[x, y] = 10**5
            l.append(np.arctan(V1[x, y]/V2[x, y]))
            
    minVal = np.min(l)
    l -= minVal
    maxVal = np.max(l)
    l /= maxVal
    l = np.reshape(l,(np.shape(V1)))
#     show(l)
    return l

def calcRmap(H, I):
    r = (H + 1)/(I + 1)
    r -= np.min(r)
    r /= np.max(r)
    return r

def piCalc(i, rmap):
    count = 0
    for item in rmap:
        if item == i:
            count+=1
    return count/len(rmap)

def W1W2(rmap, W, W1, W2, t):    
    W2 = W2-piCalc(t, rmap)
    W1 = W-W2
    return W1, W2

def threshold(rmap):
    rmap = np.reshape(rmap,(-1,1))
#     print(rmap.shape)
    rmap = (rmap*255).astype(np.int8)
    W = 0
    for i in range(256):
        W+=piCalc(i, rmap)
    w2 = W
    w1 = 0
    dmin = 10**5
    value = 0
    # print("starts")
    for T in range(256):
        m1 = 0
        m2 = 0
        for val in range(T):
            pi = piCalc(val, rmap)
            w1, w2 = W1W2(rmap, W, w1, w2, val)
            m1 += val*pi/w1
        for val2 in range(T, 256):
            pi = piCalc(val2, rmap)
            w1, w2 = W1W2(rmap, W, w1, w2, val2)
            m2 = val2*pi/w2
        value += piCalc(T, rmap)*(T-m1)**2 + piCalc(T,rmap)*(T-m2)**2
        if value<dmin:
            dmin=value
    return 1- value/255


def smap(r, th):
    l = []
    i = 0
    for x in range(np.shape(r)[0]):
        for y in range(np.shape(r)[1]):
            if r[x, y] <= th:#>>>>>>>#### INVERSION DONE ####
                l.append(1)
                i+=1
            else:
                l.append(0)
    l = np.reshape(l,(np.shape(r)))
    
    return l

def shadowMap(img, disp, lamda = lamda):

    shadowImage = np.zeros((np.shape(img)))
    for x in range(np.shape(img)[0]):
        for y in range(np.shape(img)[1]):
            if disp[x, y] == 0:
                shadowImage[x, y] = img[x, y]
            else:
                shadowImage[x, y] = lamda*img[x, y] + (1 - lamda)*disp[x, y]
    image = np.reshape(shadowImage, (np.shape(img)))
#     show(disp)
    return image

def computeRGB(img, lamda = lamda, th = th, V1 = V1, V2 = V2, I = I, T = T):

    for x in range(np.shape(img[:,:,0])[0]):
        for y in range(np.shape(img[:,:,0])[1]):
            
            I[x, y], V1[x, y], V2[x, y] = T@img[x, y]

    S = calcS(V1, V2)
    H = calcH(V1, V2)
    rMap = calcRmap(H, I)
    # th = threshold(rMap)
    sMap = smap(rMap, th)
    
    shadowImage = shadowMap(img, sMap*255)
    return shadowImage

ouput = computeRGB(img)
show(ouput)
save(ouput, f'shadow map')

#########################`````````````````````````````````````````````````````````##################################


def rgb2lab(img):
    return np.array(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))

def lab2rgb(img):
    return np.array(cv2.cvtColor(img, cv2.COLOR_LAB2RGB))

def L2Norm(x1,y1,x2,y2):
    return np.sqrt(abs((x2-x1)**2 + (y2-y1)**2))

def gaussian(x, sigma):
    return (1/(2*np.pi*(sigma**2)))*np.exp(-(x**2)/(2*(sigma**2)))

def convolution(fltr, image, row, column, FilterSize):
    return np.sum(fltr*image[row:row+FilterSize, column:column+FilterSize])

def FilterGenerator(image, row, column, sigmaS, sigmaE, FilterSize = FilterSize):
    gaussB = np.zeros((FilterSize, FilterSize))
    gaussE = np.zeros((FilterSize, FilterSize))
    for i in range(FilterSize):
        for j in range(FilterSize):
            gaussB[i, j] = gaussian(image[row+FilterSize//2, column+FilterSize//2] - image[row+i, column+j], sigmaS)
            gaussE[i, j] = gaussian(L2Norm(row+FilterSize//2, column+FilterSize//2, row+i, column+j), sigmaE)
    return gaussB, gaussE

def blFilter(img, sigmaS, sigmaE, FilterSize = FilterSize):
    
    img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    x = np.shape(img)[0]
    y = np.shape(img)[1]
    
    img0 = np.zeros((np.shape(img)[0]+2*(FilterSize//2), np.shape(img)[1]+2*(FilterSize//2)))
    img0[FilterSize//2:x+FilterSize//2, FilterSize//2:y+FilterSize//2] = img
    img = img0
    newImg = np.zeros((x, y))
    for row in range(x):
        for column in range(y):
            
            gaussian4Smoothing, gaussian4Edge = FilterGenerator(img, row, column, sigmaS, sigmaE)
            normalizingWeigth = 1/np.sum(gaussian4Smoothing * gaussian4Edge)
            
            chadFilter = normalizingWeigth * gaussian4Smoothing * gaussian4Edge
            pixelVal = convolution(chadFilter, img, row, column, FilterSize)
            newImg[row, column] = pixelVal
            
    I = newImg
    I = (I-np.min(I))/(np.max(I)-np.min(I))

    return I*255
output2 = blFilter(img, 50, 10)
show(output2)
save(output2, f'bilateral filter image')

f = cv2.bilateralFilter(img, 10, 50, 10)
show(f)
save(f, f'bilateral filter image using openCV')


##################################~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#################################


sobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
FilterSize = 3

def sobelConv(img, sobelX, sobelY, FilterSize = 3):
    img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    x = np.shape(img)[0]
    y = np.shape(img)[1]
    
    img0 = np.zeros((np.shape(img)[0]+2*(FilterSize//2), np.shape(img)[1]+2*(FilterSize//2)))
    img0[FilterSize//2:x+FilterSize//2, FilterSize//2:y+FilterSize//2] = img
    img = img0

    newImgX = np.zeros((x, y))
    newImgY = np.zeros((x, y))
    for row in range(x):
        for column in range(y):
            pixelValY = convolution(sobelY, img, row, column, FilterSize)
            pixelValX = convolution(sobelX, img, row, column, FilterSize)
            newImgY[row, column] = pixelValY
            newImgX[row, column] = pixelValX
            
    Ix = newImgX
    Iy = newImgY
    Ix = (Ix-np.min(Ix))/(np.max(Ix)-np.min(Ix))
    Iy = (Iy-np.min(Iy))/(np.max(Iy)-np.min(Iy))
    return Ix*255, Iy*255

sx, sy = sobelConv(img, sobelX, sobelY)

edgeMap = np.sqrt(sx**2 + sy**2)
edgeMap = edgeMap**2.5
edgeMap = (edgeMap-np.min(edgeMap))/(np.max(edgeMap)-np.min(edgeMap))*255

show(edgeMap)
save(edgeMap, f'edgeMap image')


lineDraft = np.ones((np.shape(edgeMap)))*255

for i in range(np.shape(edgeMap)[0]):
    for j in range(np.shape(edgeMap)[1]):
        lineDraft[i, j] = edgeMap[i, j] < 65

ld = lineDraft*255
show(ld)
save(ld, f'line draft image')

#############################~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#####################################


lab = rgb2lab(img)

lab[:,:,1] = lab[:,:,1] +(np.min(lab[:,:,1]) + np.max(lab[:,:,1]))/2
lab[:,:,2] = lab[:,:,2] +(np.min(lab[:,:,2]) + np.max(lab[:,:,2]))/2

cm = lab2rgb(lab)
shadowimg = computeRGB(img)

sI1 = shadowimg * (1 + np.tanh(ro*(cm - 128)))/2
show(sI1)
save(sI1, f'saturation corrected image')


def ARI(sI1, linedraft):
    beta = .75
    linedraft *= 255
    val = np.zeros(np.shape(sI1))
    for i in range(np.shape(sI1)[0]):
        for j in range(np.shape(sI1)[1]):
            for k in range(np.shape(sI1)[2]):
                if linedraft[i,j] == 0:
                    val[i,j,k] = beta * sI1[i,j,k] + (1-beta) * linedraft[i,j]
                else:
                    val[i,j,k] = sI1[i,j,k]
                    
    return val

artistImage = ARI(sI1, lineDraft)
show(artistImage)
# k = Image.fromarray(artistImage.astype(np.uint8))
save(artistImage, f'ARI')


# In[ ]:




