#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU
from keras.layers import Conv2DTranspose, Dropout, ReLU, Input, Concatenate, ZeroPadding2D
from keras.optimizers import Adam
from keras.utils import plot_model


# In[ ]:


import os
from os import listdir
from PIL import Image

pth = "/kaggle/input/dipdataset/seam_carving_input"
# p = "C:\\Users\\GAGANDEEP SINGH\\Downloads\\"
images = os.listdir(pth)


# In[ ]:


devices = tf.config.experimental.list_physical_devices("GPU")
# BATCH_SIZE = 32


# In[ ]:


def load(image_file):
    path = "/kaggle/input/dipdataset/seam_carving_input/"
    input_image = []
    real_image = []
    for i in image_file:
#         print(i)
        newPath = path + str(i)
#         print(newPath)
        imageI = tf.io.read_file(newPath)
        imageI = tf.io.decode_jpeg(imageI, channels = 3)
#         print((image_file))
        pth = "/kaggle/input/dipdataset/output_main/"
        adres = pth + str(i)[:-4] + "_out.jpg"
#         print(adres)
        image_fileO = adres
#         print(image_fileO)
        imageO = Image.open(image_fileO)
        imageO = np.array(imageO)

        real_image.append(imageI[:, :, :])
        input_image.append(imageO[:, :, :])
    
#     input_image = tf.cast(input_image, tf.float32)
#     real_image = tf.cast(real_image, tf.float32)
    input_image = np.reshape(input_image,(-1,384,384,3))
    real_image = np.reshape(real_image,(-1,512,512,3))
    return np.array(input_image, dtype = np.float32), np.array(real_image, dtype = np.float32)
y, x = load(images)
print(np.max(x))
print(np.max(y))
x = x/np.max(x)
y = y/np.max(y)
# dataset = list(zip(x,y))


# In[ ]:


# downsample block
def downsample(filters, size, batchnorm = True):
    init = tf.random_normal_initializer(0.,0.02)
    result = Sequential()
    result.add(Conv2D(filters, size, strides = 2, padding = "same", kernel_initializer = init, use_bias = False))
    if batchnorm == True:
        result.add(BatchNormalization())
        
    result.add(LeakyReLU())
    return result
down_model = downsample(3,4)
down_result = down_model(tf.expand_dims(x, axis = 0))
print(down_result.shape)


# In[ ]:


# upsample block
down_result = tf.reshape(down_result, (-1, down_result.shape[2], down_result.shape[3], down_result.shape[4]))
def upsample(filters, size, dropout = False):
    init = tf.random_normal_initializer(0, 0.02)
    result = Sequential()
    result.add(Conv2DTranspose(filters, size, strides = 2, padding = "same", kernel_initializer = init, use_bias = False))
    result.add(BatchNormalization())
    if dropout == True:
        result.add(Dropout(0.5))
    result.add(ReLU())
    return result
up_model = upsample(3,4)
up_result = up_model(down_result)
print(up_result.shape)


# In[ ]:


def generator():
    
    inputs = Input(shape = [512, 512, 3])
#     print(f"gen {np.shape(inputs)}")
    down_stack = [
        downsample(64, 4, batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4)
    ]
    
    
    up_stack = [
        upsample(512, 4, dropout=True),
        upsample(512, 4, dropout=True),
        upsample(512, 4),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
#         upsample(64, 4),
    ]
    init = tf.random_normal_initializer(0., 0.02)
    last = Conv2DTranspose(3, 4, strides = 3, padding = "same", kernel_initializer = init, activation ="tanh")
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = Concatenate()([x, skip])
    
    x = last(x)
    print(f"gen {np.shape(inputs), (np.shape(x))}")
    return Model(inputs = inputs, outputs = x)


# In[ ]:


gen = generator()
gen.summary()

LAMBDA = 100
plot_model(gen, show_shapes=True, dpi = 64)

from keras.losses import BinaryCrossentropy
loss_function = BinaryCrossentropy()


# In[ ]:


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_function(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
#     print(f"gen loss is {gan_loss, LAMBDA * l1_loss}")
    return total_gen_loss, gan_loss, l1_loss


# In[ ]:


def discriminator():
    init = tf.random_normal_initializer(0., 0.02)
    
    inp = Input(shape = [384, 384, 3], name = "targetA")
    tar = Input(shape = [384, 384, 3], name = "targetB")
    x = Concatenate()([inp, tar])
    down1 = downsample(64,4,False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)
    
    zero_pad1 = ZeroPadding2D()(down3)
    conv = Conv2D(256, 4, strides = 1, kernel_initializer = init, use_bias = False)(zero_pad1)
    leaky_relu = LeakyReLU()(conv)
    zero_pad2 = ZeroPadding2D()(leaky_relu)
    last = Conv2D(1, 4, strides = 1, kernel_initializer=init)(zero_pad2)
#     last = Dense(1, activation = "sigmoid")(last)
    return Model(inputs = [inp, tar], outputs = last)


# In[ ]:


disc = discriminator()
disc.summary()
plot_model(disc, show_shapes=True, dpi = 64)


# In[ ]:


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_function(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_function(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
#     print(f"desc loss is {real_loss, generated_loss}")
    return total_disc_loss


# In[ ]:


generator_optimizer = Adam(lr= 1e-4, beta_1=0.5)
discriminator_optimizer = Adam(lr = 1e-4, beta_1=0.5)


# In[ ]:


def save_images(model, test_input, target, epoch):
    
    prediction = model(test_input, training= True)
    prediction = np.reshape(prediction,(384,384,3))
    test_input = np.reshape(test_input,(512,512,3))
    target = np.reshape(target,(384,384,3))
    plt.figure(figsize = (15,15))
    display_list= [test_input, target, prediction]
    title = ["Input Image", "Ground Truth", "Predicton Image"]
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis("off")
    plt.savefig(f"/kaggle/working/epoch_{epoch}.jpg")
    plt.close()
    
    
# make sure output directory exists to save images
if os.path.exists("output"):
    os.mkdir("output")
    
epochs = 300


# In[ ]:


@tf.function
def train_step(input_image, target, epoch):
#     print("chad")
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = gen(input_image, training = True)
#         targetA = target
#         targetB = target
        disc_real_output = disc([target, target], training = True)
    
        disc_generated_output = disc([target, gen_output], training = True)
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        generator_gradients = gen_tape.gradient(gen_total_loss, gen.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)
        
        generator_optimizer.apply_gradients(zip(generator_gradients, gen.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))
        return gen_total_loss, disc_loss


# In[ ]:



def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()
        for input_, target in test_ds.take(1):
            input_ = tf.expand_dims(input_, axis=0)
            save_images(gen, input_, target, epoch)
            
        # Train
        print(f"Epoch {epoch}")
        for n, (input_, target) in train_ds.enumerate():
            input_ = tf.expand_dims(input_, axis=0)
            target = tf.expand_dims(target, axis=0)

            gen_loss, disc_loss = train_step(input_, target, epoch)
        print("Generator loss {:.2f} Discriminator loss {}".format(gen_loss, disc_loss))
        print("Time take for epoch {} is {} sec\n".format(epoch+1, time.time() - start))
        


# In[ ]:


import keras
keras.backend.clear_session()
dataset = tf.data.Dataset.from_tensor_slices((x, y))
datasetTest = tf.data.Dataset.from_tensor_slices((x[5:10,:,:,:], y[5:10,:,:,:]))


BATCH_SIZE = 32

# Assuming you have already created your dataset
# iterator = iter(datasetTest)
# # Loop through the dataset and print the elements
# i = 0
# for element in iterator:
#     i+=1
# #     print("hi")
#     input_image, output_image = element
    
#     fig, axes = plt.subplots(1,2, figsize = (16,5))
#     axes[0].imshow(input_image)
#     axes[1].imshow(output_image)
#     if i==4:
#         break
#     print("Input Image Shape:", input_image.shape)
#     print("Output Image Shape:", output_image.shape)


# In[ ]:


fit(dataset, epochs, datasetTest)


# In[ ]:


path = "/kaggle/input/dipdataset/seam_carving_input/"
input_image = []
real_image = []
# for i in image_file:
#         print(i)
i = "1007.jpg"
newPath = path + str(i)
#         print(newPath)
imageI = tf.io.read_file(newPath)
imageI = tf.io.decode_jpeg(imageI, channels = 3)
im = np.array(imageI)
im = np.reshape(im,(1,512,512,3))/255

a = gen(im, training = False)
# im.show(a)
a = np.reshape(a,(384,384,3))
# print(a)
def show(array):
    array = np.array(array)#/np.max(array)*255
    data = Image.fromarray(array.astype(np.uint8))
    data.show()
    
# show(a)
fig, axes = plt.subplots(1,2, figsize = (16,5))
axes[0].imshow(imageI)
axes[1].imshow(a)


# In[ ]:


gen.save("seamCarving")


# In[ ]:




