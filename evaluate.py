import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import cv2
import numpy as np
import random
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, GRU, Bidirectional, TimeDistributed
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Activation, BatchNormalization, Subtract, Multiply, Add, Concatenate
import pywt
from pywt import dwt2,idwt2
import PIL.Image as Image

#BSDS100 Dataset
#TESTDIR ="/home/zeshanfayyaz/LIA/STRIPE_DATASETS/BSDS100/"

#Infrared100 Dataset
#TESTDIR ="/home/zeshanfayyaz/LIA/STRIPE_DATASETS/INFRARED100/"

#Set12 Dataset
#TESTDIR ="/home/zeshanfayyaz/LIA/STRIPE_DATASETS/Set12/"

#Linnaeus 5 Dataset
#TESTDIR ="/home/zeshanfayyaz/LIA/STRIPE_DATASETS/Linnaeus5/"

#Urban100 Dataset
#TESTDIR ="/home/zeshanfayyaz/LIA/STRIPE_DATASETS/Urban100/"

MODEL = "/home/zeshanfayyaz/LIA/NoiseRemoval/BETA/15GRU/destriping_model.h5"
saved_model = load_model(MODEL)
saved_model.summary()

def psnr_ssim_metrics(ground_truth, predicted, degraded):
    # PSNR
    psnr_degraded = peak_signal_noise_ratio(ground_truth, degraded)
    psnr_predicted = peak_signal_noise_ratio(ground_truth, predicted)
    psnr_difference = psnr_predicted - psnr_degraded
    # SSIM
    ssim_degraded = structural_similarity(ground_truth, degraded)
    ssim_predicted = structural_similarity(ground_truth, predicted)
    ssim_difference = ssim_predicted - ssim_degraded
    return psnr_degraded, psnr_predicted, psnr_difference, ssim_degraded, ssim_predicted, ssim_difference

def average(lst):
    return sum(lst) / len(lst)

def Addnoise(image,beta= (255*0.15)):
    image.astype('float32') 
    np.random.seed(0)
    G_col =  np.random.normal(0, beta, image.shape[1])
    G_noise = np.tile(G_col,(image.shape[0],1))
    G_noise = np.reshape(G_noise,image.shape)

    image_G = image + G_noise
    return image_G

print("Creating Testing Data...")
testing_data = []
IMG_SIZE = 256

def create_testing_data():
    for img in os.listdir(TESTDIR):
        try: 
            img_array = cv2.imread(os.path.join(TESTDIR,img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            testing_data.append([new_array])
        except Exception as e:
            pass 

create_testing_data()  

testing_data = np.array(testing_data)
print("Testing Data Length: " + str(len(testing_data)))

print("Degrading Testing Data...")
model_testing_data = []
clean_testing_data = []
for images in testing_data:
    model_image = Addnoise(images[0])
    model_testing_data.append(model_image/255.0)
    clean_testing_data.append(images.squeeze()/255.0)
clean_testing_data = np.array(clean_testing_data)
model_testing_data = np.array(model_testing_data)
print("Degrading Testing Data... Done")

output = saved_model.predict(model_testing_data)

psnr_degraded_lst = []
psnr_predicted_lst = []
psnr_difference_lst = []
ssim_degraded_lst = []
ssim_predicted_lst = []
ssim_difference_lst = []

for i in range(len(output)):
    psnr_degraded, psnr_predicted, psnr_difference, ssim_degraded, ssim_predicted, ssim_difference = \
        psnr_ssim_metrics(clean_testing_data[i], output[i], model_testing_data[i])
    psnr_degraded_lst.append(psnr_degraded)
    psnr_predicted_lst.append(psnr_predicted)
    psnr_difference_lst.append(psnr_difference)
    ssim_degraded_lst.append(ssim_degraded)
    ssim_predicted_lst.append(ssim_predicted)
    ssim_difference_lst.append(ssim_difference)

psnr_degraded_average = average(psnr_degraded_lst)
print("The average PSNR of all Test Degraded Images w.r.t Ground Truth: " + str(psnr_degraded_average))
psnr_predicted_average = average(psnr_predicted_lst)
print("The average PSNR of all Test Predicted Images w.r.t Ground Truth: " + str(psnr_predicted_average))
psnr_difference_average = average(psnr_difference_lst)
print("The average PSNR Difference: " + str(psnr_difference_average))

ssim_degraded_average = average(ssim_degraded_lst)
print("The average SSIM of all Test Degraded Images w.r.t Ground Truth: " + str(ssim_degraded_average))
ssim_predicted_average = average(ssim_predicted_lst)
print("The average SSIM of all Test Predicted Images w.r.t Ground Truth: " + str(ssim_predicted_average))
ssim_difference_average = average(ssim_difference_lst)
print("The average SSIM Difference: " + str(ssim_difference_average))


