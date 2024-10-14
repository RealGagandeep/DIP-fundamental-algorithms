#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from os import listdir


gen = tf.keras.models.load_model("seamCarving", compile=False)

# get the path/directory
pth = "/kaggle/input/dipdataset/seam_carving_input/"
p = "/kaggle/input/dipdataset/output_main/"

rmsVal = []
ssimVal = []
nmiVal = []
for images in os.listdir(pth):
    # check if the image ends with png
    if (images.endswith(".jpg")):
        nme = f"{images}"[:-4]
#         print(nme)
        name = nme + ".jpg"
        imgPath = pth + name
        img = Image.open(imgPath)
        ground_truth = np.array(img)
        ground_truth = np.reshape(ground_truth,(1,512,512,3))
        ground_truth = gen(ground_truth)
        ground_truth = np.array(ground_truth)
        ground_truth = np.reshape(ground_truth,(-1))
#         ground_truth = np.sum(ground_truth, axis = 2)
        ground_truth = np.array(ground_truth)

        chadNme = p + nme + "_out.jpg"
        pred = Image.open(chadNme)
        predicted_image = np.array(pred)
        predicted_image = np.reshape(predicted_image,(-1))

        predicted_image = np.array(predicted_image)


        # Compute Root Mean Squared Error (RMSE)
        rmse = np.sqrt(((ground_truth - predicted_image) ** 2).mean())

        # Compute Normalized Mutual Information (NMI)
        nmi_value = nmi(ground_truth, predicted_image)

        # Compute Structural Similarity Index Metric (SSIM)
        ssim_value, _ = ssim(ground_truth, predicted_image, full=True, data_range = 1000.00000)

        # Print the computed metrics
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Normalized Mutual Information (NMI): {nmi_value}")
        print(f"Structural Similarity Index Metric (SSIM): {ssim_value}")
        rmsVal.append(rmse)
        ssimVal.append(ssim_value)
        nmiVal.append(nmi_value)

        # Additional metric - Peak Signal-to-Noise Ratio (PSNR)
#         psnr_value = psnr(ground_truth, predicted_image)
#         print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr_value}")

