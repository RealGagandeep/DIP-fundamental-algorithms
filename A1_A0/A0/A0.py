#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import sys
import cv2
from PIL import Image
def save(array, name):
    k = Image.fromarray(array.astype(np.uint8))
    k.save(f'{name}.jpg')
    
def show(array):
    array = np.array(array)/np.max(array)*255
    data = Image.fromarray(array.astype(np.uint8))
    data.show()

def biInterpolation(img, newH, newW):
    oldH, oldW, _ = np.shape(img)
    ratioY = oldH / newH
    ratioX = oldW / newW

    newImg = np.zeros((newH, newW, 3))

    for y in range(newH):
        for x in range(newW):
            oldX = x * ratioX
            oldY = y * ratioY

            x0 = int(oldX)
            y0 = int(oldY)
            x1 = min(x0 + 1, oldW - 1)
            y1 = min(y0 + 1, oldH - 1)

            xdiff = oldX - x0
            ydiff = oldY - y0

            for channel in range(3):
                newImg[y, x, channel] = (img[y0, x0, channel]*(1 - xdiff)*(1 - ydiff) + img[y0, x1, channel]*xdiff*(1 - ydiff) +
                                        img[y1, x0, channel]*(1 - xdiff)*ydiff +
                                        img[y1, x1, channel]*xdiff*ydiff)

    return newImg

# img = Image.open('C:\\Users\\GAGANDEEP SINGH\\Desktop\\2.jpg')
img = Image.open(sys.argv[1])
image = np.array(img)

BiImg = biInterpolation(image, 200, 200)

# show(BiImg)
save(BiImg, f'bilateral interpolation Image')


# In[ ]:




