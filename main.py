import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Bidirectional, TimeDistributed
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def Degrade(image, noise_val):
    beta = noise_val * np.random.rand(1)
    image.astype('float32')
    np.random.seed(0)
    G_col = np.random.normal(0, beta, image.shape[1])
    G_noise = np.tile(G_col, (image.shape[0], 1))
    G_noise = np.reshape(G_noise, image.shape)

    image_G = image + G_noise
    return image_G, G_noise


DATADIR = "/home/dplatnick/research/codetest2/cifar10traindir"
TESTDIR = "/home/dplatnick/research/codetest2/cifar10testdir"

print("Creating Training and Validation Data...")
training_validation_data = []
training_data = []
validation_data = []
IMG_SIZE = 32
CATEGORIES = ["Clean", "Striped"]


def create_training_data():
    class_num = CATEGORIES.index("Clean")
    for img in os.listdir(DATADIR):
        try:
            img_array = cv2.imread(os.path.join(DATADIR, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_validation_data.append([new_array, class_num])
            random.shuffle(training_data)
        except Exception as e:
            pass


create_training_data()
print("Total Training + Validation Length: " + str(len(training_validation_data)))
print("Using 85% Train and 15% Validation...")
numtosplit = int(0.85 * (len(training_validation_data)))
training_data = training_validation_data[:numtosplit]
validation_data = training_validation_data[numtosplit:]

print("Training Data Length: " + str(len(training_data)))
print("Validation Data Length: " + str(len(validation_data)))

print("Creating Testing Data...")
testing_data = []


def create_testing_data():
    for img in os.listdir(TESTDIR):
        try:
            img_array = cv2.imread(os.path.join(TESTDIR, img), cv2.IMREAD_GRAYSCALE)
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
    model_image = Degrade(images[0], 25)[0]
    model_testing_data.append(model_image / 255.0)
    clean_testing_data.append(images.squeeze() / 255.0)
clean_testing_data = np.array(clean_testing_data)
model_testing_data = np.array(model_testing_data)
print("Degrading Testing Data... Done")

num_epochs = 100
loss_train = []
loss_val = []
X_train = []
z_train = []

#Num of epochs
for i in range(3):
    X = []
    z = []
    X_validation = []
    z_validation = []
    for j in range(10):
        for images, label in training_data:
            Degraded = Degrade(images, 25)
            X.append(Degraded[0] / 255.0)
            z.append(images / 255.0)

        for images, label in validation_data:
            Degraded = Degrade(images, 25)
            X_validation.append(Degraded[0] / 255.0)
            z_validation.append(images / 255.0)


        print("Done Adding Noise Instance: " + str(j+1))

    print("Adding Noise Instances... Done")

    print("Reshaping Arrays..")

    X_train = X[0:2]
    z_train = z[0:2]
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE)
    X_validation = np.array(X_validation).reshape(-1, IMG_SIZE, IMG_SIZE)
    X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE)
    X = np.array(X)
    z = np.array(z)
    X_validation = np.array(X_validation)
    z_validation = np.array(z_validation)
    X_train = np.array(X_train)
    z_train = np.array(z_train)

    print("Reshaping Arrays... Done")

    inputs = Input(shape=(32, 32))
    inputs_t = tf.transpose(inputs, perm=[0, 2, 1])

    # Layer 1
    output_1 = (Bidirectional(GRU(32, return_sequences=True), merge_mode="ave"))(inputs_t)
    # output_1 = TimeDistributed(Dense(32))(output_1)
    input_2 = tf.keras.layers.Subtract()([inputs_t, output_1])
    # Layer 2
    output_2 = (Bidirectional(GRU(32, return_sequences=True), merge_mode="ave"))(input_2)
    # output_2 = TimeDistributed(Dense(32))(output_2)
    input_3 = keras.layers.Subtract()([input_2, output_2])
    # layer 3
    output_3 = (Bidirectional(GRU(32, return_sequences=True), merge_mode="ave"))(input_3)
    # output_3 = TimeDistributed(Dense(32))(output_3)
    input_4 = keras.layers.Subtract()([input_3, output_3])
    # Layer 4
    output_4 = (Bidirectional(GRU(32, return_sequences=True), merge_mode="ave"))(input_4)
    # output_4 = TimeDistributed(Dense(32))(output_4)
    input_5 = keras.layers.Subtract()([input_4, output_4])
    # Layer 5
    output_5 = (Bidirectional(GRU(32, return_sequences=True), merge_mode="ave"))(input_5)
    # output_5 = TimeDistributed(Dense(32))(output_5)
    input_6 = keras.layers.Subtract()([input_5, output_5])

    output_6 = Bidirectional(GRU(32, return_sequences=True), merge_mode="ave")(input_6)
    output_GRU = TimeDistributed(Dense(32))(output_6)

    output_GRU = tf.transpose(output_GRU, perm=[0, 2, 1])
    real_output = tf.keras.layers.Subtract()([inputs, output_GRU])

    model = Model(inputs=inputs, outputs=real_output)
    model.summary()
    print("Model Compiled")

    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

    model.compile(
        optimizer=opt,
        loss='mean_squared_error',
        metrics=['mse']
    )
    # Then, train the model with fit()
    history = model.fit(X, z,
                        batch_size=100,
                        epochs=1,
                        validation_data=(X_validation, z_validation)
                        )
    loss_train.append(history.history['loss'])
    loss_val.append(history.history['val_loss'])
    print("Done Epoch: " + str(i+1))

plt.plot(loss_train)
plt.title("Training Loss")
plt.grid()
plt.show()

plt.plot(loss_val)
plt.title("Validation Loss")
plt.grid()
plt.show()

epoch_list = []
for i in range(1, 101):
    epoch_list.append(i)

plt.plot(epoch_list, loss_val, label = 'val loss')
plt.plot(epoch_list, loss_train, label = 'train loss')
plt.title("Training Loss and Validation Loss")
plt.legend(loc='center right')
plt.grid()
plt.show()

output_test = model.predict(model_testing_data)

print("Test Image Sample...")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(11, 11))
# Plot 1: Clean Testing Image
ax1.imshow(clean_testing_data[0], cmap="gray")
ax1.title.set_text('Ground Truth')

# Plot 2: Degraded Testing Image
model_testing_data = np.array(model_testing_data)
ax2.imshow(model_testing_data[0], cmap="gray")
ax2.title.set_text('Degraded Image')

# Plot 3: Predicted Clean Image
ax3.imshow(output_test[0], cmap="gray")
ax3.title.set_text('Predicted Image')
fig.suptitle("Testing Image")
fig.show()

print(len(X_train))
print(X_train[0].shape)
print(len(X))
print(X[0].shape)

#Comparing PSNR and SSIM
#Compare PSNR where Img_True = Clean Testing Data and Img_Test = Degraded Original Testing Data
#Then, compare PSNR2 where Img_True = Clean Image and Img_Test = Ouput of Network
#We want to have it so that PSNR2 > PSNR. This indicates that the PSNR of the PREDICTED image is greater than the PSNR of the ORIGINAL image

PSNR = peak_signal_noise_ratio(clean_testing_data[0], model_testing_data[0])
print("PSNR of Original Degraded Image: " + str(PSNR))
PSNR2 = peak_signal_noise_ratio(clean_testing_data[0], output_test[0])
print("PSNR of Predicted Clean Image: " + str(PSNR2))
print("\nIf the calculated difference is positive, our predicted image is of better quality than degraded")
print("If the calculated difference is negative, our predicted image is of worse quality than degraded")
print("\nDifference: " + str(PSNR2 - PSNR))

#Here, we compare the SSIM between (a) the GroundTruth and the NoisyImage. And (b) the GroundTruth and the PredictedImage
#Again, we aim to have SSIM2 > SSIM. This indicates that the SSIM of the PREDICTED image is greater than the SSIM of the ORIGINAL

SSIM = structural_similarity(clean_testing_data[0], model_testing_data[0])
print("SSIM of Original Degraded Image in Reference to Ground Truth: " + str(SSIM))
SSIM2 = structural_similarity(clean_testing_data[0], output_test[0])
print("SSIM of Predicted Clean Image in Reference to Ground Truth: " + str(SSIM2))
print("Difference: " + str(SSIM2 - SSIM))

#We can perform the same operation as above, but this time to one of the TRAINING images
output_train = model.predict(X_train)

print("Train Image Sample...")

fig2, (ax1t, ax2t, ax3t) = plt.subplots(1, 3, sharey=True, figsize=(11, 11))
# Plot 1: Clean Testing Image
ax1t.imshow(z_train[0], cmap="gray")
ax1t.title.set_text('Ground Truth')

# Plot 2: Degraded Testing Image
model_testing_data = np.array(model_testing_data)
ax2t.imshow(X_train[0], cmap="gray")
ax2t.title.set_text('Degraded Image')

# Plot 3: Predicted Clean Image
ax3t.imshow(output_train[0], cmap="gray")
ax3t.title.set_text('Predicted Image')
fig2.suptitle("Training Image")
fig2.show()

SSIM_train = structural_similarity(z[0], X[0])
print("SSIM of Degraded Training Image in Reference to Ground Truth: " + str(SSIM_train))
SSIM2_train = structural_similarity(z[0], output_train[0])
print("SSIM of Predicted Clean Image in Referen ce to Ground Truth: " + str(SSIM2_train))
print("Difference: " + str(SSIM2 - SSIM_train))

PSNR_train = peak_signal_noise_ratio(z[0], X[0])
print("PSNR of Degraded Training Image: " + str(PSNR_train))
PSNR2_train = peak_signal_noise_ratio(z[0], output_train[0])
print("PSNR of Predicted Clean Image: " + str(PSNR2_train))
print("Difference: " + str(PSNR2_train - PSNR_train))
