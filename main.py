import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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


# Degrade Function which takes the clean image as input and returns the noisy image and the noise itself. Same dimensionality, I.e. (256,256)
def Degrade(image, noise_val):
    beta = noise_val * np.random.rand(1)
    image.astype('float32')
    np.random.seed(0)
    G_col = np.random.normal(0, beta, image.shape[1])
    G_noise = np.tile(G_col, (image.shape[0], 1))
    G_noise = np.reshape(G_noise, image.shape)

    image_G = image + G_noise
    return image_G, G_noise


# The paths to the image locations
# DATADIR refers to directory that contains the clean images for TRAINING
# TESTDIR refers to the directory that contains the clean images for TRAINING
DATADIR = "/home/zeshanfayyaz/LIA/Local_Images/Train/"
TESTDIR = "/home/zeshanfayyaz/LIA/Local_Images/Test/"

# We seperate the DATADIR images into Training Data and Validation Data
print("Creating Training and Validation Data...")
training_validation_data = []
training_data = []
validation_data = []
IMG_SIZE = 256
# Although we label the images as '0' (Clean) this is not needed
CATEGORIES = ["Clean", "Striped"]


# The Training + Validation Data is all the images in the DATADIR directory, resized to IMG_SIZE, IMG_SIZE
def create_training_data():
    class_num = CATEGORIES.index("Clean")
    for img in os.listdir(DATADIR):
        try:
            img_array = cv2.imread(os.path.join(DATADIR, img), cv2.IMREAD_GRAYSCALE)
            # Resizing the Images to IMG_SIZE,IMG_SIZE
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_validation_data.append([new_array, class_num])
            # Shuffling the data
            random.shuffle(training_data)
        # Pass exception in case we are unable to read an image due to corruption, ignore that image
        except Exception as e:
            pass


create_training_data()

# Total Amount of Images we have for Training + Validation
print("Total Training + Validation Length: " + str(len(training_validation_data)))
print("Using 85% Train and 15% Validation...")

# We can decide what split we want for Validation. In this case, 85% of DATADIR Images will go towards Training
# And the remaining 15% will go towards Validation
numtosplit = int(0.85 * (len(training_validation_data)))
training_data = training_validation_data[:numtosplit]
validation_data = training_validation_data[numtosplit:]

print("Training Data Length: " + str(len(training_data)))
print("Validation Data Length: " + str(len(validation_data)))

# We can perform the same operation on Testing Data
print("Creating Testing Data...")
testing_data = []


def create_testing_data():
    for img in os.listdir(TESTDIR):
        try:
            img_array = cv2.imread(os.path.join(TESTDIR, img), cv2.IMREAD_GRAYSCALE)
            # Resize the images in TESTDIR to IMG_SIZE, IMG_SIZE
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            testing_data.append([new_array])
        # Pass exception in case we are unable to read an image due to corruption, ignore that image
        except Exception as e:
            pass


create_testing_data()

testing_data = np.array(testing_data)
print("Testing Data Length: " + str(len(testing_data)))

# Degrade the Testing Data (we use this later when we want to evaluate our model on Testing Images)
# The dirty Testing images are stored in "model_testing_data"
# The clean Testing images are stored in "clean_testing_data"
print("Degrading Testing Data...")
model_testing_data = []
clean_testing_data = []
for images in testing_data:
    # We degrade the Test images with a value of 25
    model_image = Degrade(images[0], 25)[0]
    # We normalize all pixel values (max 255.0)
    model_testing_data.append(model_image / 255.0)
    clean_testing_data.append(images.squeeze() / 255.0)
clean_testing_data = np.array(clean_testing_data)
model_testing_data = np.array(model_testing_data)
print("Degrading Testing Data... Done")

# Set the number of epochs to 100
# In this case, X is our input and z is our target - as opposed to the conventional X and y
num_epochs = 100
loss_train = []
loss_val = []
X_debug = []
z_debug = []

# Num of epochs

inputs = Input(shape=(256, 256))
# Transpose the input shape replacing rows with columns and columns with rows
inputs_t = tf.transpose(inputs, perm=[0, 2, 1])

# We perform this manually for ease of visualization, however can be done in a for loop
# MODEL 3
# Layer 1
# Return sequences is TRUE as we want an output for every timestep, and not a "many-to-one" output
# Merge_mode is set to AVERAGE - in order to maintain dimensionality (256,256) [default is CONCAT]
output_1 = (Bidirectional(GRU(256, return_sequences=True), merge_mode="ave"))(inputs_t)
# The input of layer 2 is the output of layer 0 [transpose] SUBTRACT the output of layer 1
input_2 = tf.keras.layers.Subtract()([inputs_t, output_1])
# Layer 2
output_2 = (Bidirectional(GRU(256, return_sequences=True), merge_mode="ave"))(input_2)
# The input of layer 3 is the input of layer 2 SUBTRACT the output of layer 2
input_3 = keras.layers.Subtract()([input_2, output_2])
# layer 3
output_3 = (Bidirectional(GRU(256, return_sequences=True), merge_mode="ave"))(input_3)
# The input of layer 4 is the input of layer 3 SUBTRACT the output of layer 3
input_4 = keras.layers.Subtract()([input_3, output_3])
# Layer 4
output_4 = (Bidirectional(GRU(256, return_sequences=True), merge_mode="ave"))(input_4)
# The input of layer 5 is the input of layer 4 SUBTRACT the output of layer 4
input_5 = keras.layers.Subtract()([input_4, output_4])
# Layer 5
output_5 = (Bidirectional(GRU(256, return_sequences=True), merge_mode="ave"))(input_5)
# The input of layer 6 is the input of layer 5 SUBTRACT the output of layer 5
input_6 = keras.layers.Subtract()([input_5, output_5])
# Layer 6
output_6 = Bidirectional(GRU(256, return_sequences=True), merge_mode="ave")(input_6)
# Perform TimeDistributed Operation to final output of GRU
# Performs operation on each temporal slice of output
output_GRU = TimeDistributed(Dense(256))(output_6)

# Transpose the image once again, giving us original dimensionality
output_GRU = tf.transpose(output_GRU, perm=[0, 2, 1])
# We aim for the real_output to be as close as possible to 'z' -> the Clean Image
# The clean image is the dirty image [Input] SUBTRACT the noise [Output of the Network (output_GRU)]
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
for i in range(num_epochs):
    X = []
    z = []
    X_validation = []
    z_validation = []
    # We must degrade our training noise with random instances of stripe noise
    # In this case, for each of the 100 epochs, we add 10 instances of stripe noise to Training Data AND Validation Data
    # The instances added to Training AND Validation are not correlated

    for images, label in training_data:
        noisy_image, _ = Degrade(images, 25)
        # We append the training images to X and z
        X.append(noisy_image / 255.0)  # X is the Dirty Training Images
        z.append(images / 255.0)  # z is the Clean Training Images [Target]

    for images, label in validation_data:
        noisy_image, _ = Degrade(images, 25)
        X_validation.append(noisy_image / 255.0)  # X_validation is the Dirty Validation Images
        z_validation.append(images / 255.0)  # z_validation is the Clean Validation Images [Target]

    print("Adding Noise Instances... Done")

    print("Reshaping Arrays..")

    # Assume we have 1000 Training Images, adding 10 instances of stripe noise creates 10,000 Images
    # I do not care about all these images, I just want to extract a very small subset of these Images so I can evaluate them later
    # In this case, for every epoch, I just extract the first 3 Images of X and z
    # Place them into X_debug and z_debug so I can test them later
    X_debug = X[0:2]
    z_debug = z[0:2]

    # We can reshape the arrays as well as convert them to numpy arrays
    #X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE)
    #X_validation = np.array(X_validation).reshape(-1, IMG_SIZE, IMG_SIZE)
    #X_debug = np.array(X_debug).reshape(-1, IMG_SIZE, IMG_SIZE)
    X = np.array(X)
    z = np.array(z)
    X_validation = np.array(X_validation)
    z_validation = np.array(z_validation)
    X_debug = np.array(X_debug)
    z_debug = np.array(z_debug)

    print("Reshaping Arrays... Done")

    # The input shape refers to the size of the images used
    # CIFAR10 Image Shape: (32,32)
    # Natural_Dataset Image Shape: (256,256)

    
    # Then, train the model with fit()
    # We specify the batch size
    # Epochs = 1 due to it being in a for loop
    history = model.fit(X, z,
                        batch_size=100,
                        epochs=1,
                        # We manually set the validation data as defined at the top
                        validation_data=(X_validation, z_validation)
                        )
    # We append the metrics to their respective lists
    loss_train.append(history.history['loss'])
    loss_val.append(history.history['val_loss'])
    print("Done Epoch: " + str(i + 1))

# Once training is complete, plot the Training Loss
plt.plot(loss_train)
plt.title("Training Loss")
plt.grid()
plt.show()

# Once training is complete, plot the Validation Loss
plt.plot(loss_val)
plt.title("Validation Loss")
plt.grid()
plt.show()

# Once training is complete, plot the Training and Validation Loss on the same axis
# Define range of x axis
epoch_list = []
for i in range(1, 201):
    epoch_list.append(i)

plt.plot(epoch_list, loss_val, label='val loss')
plt.plot(epoch_list, loss_train, label='train loss')
plt.title("Training Loss and Validation Loss")
plt.legend(loc='center right')
plt.grid()
plt.show()

# Test out network using the CLEAN TEST Images, as we defined at the top of the code
output_test = model.predict(model_testing_data)

# We print an example image of output_test so we can visually inspect any loss of stripe noise
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

# Comparing PSNR and SSIM
# Compare PSNR where Img_True = Clean Testing Data and Img_Test = Degraded Original Testing Data
# Then, compare PSNR2 where Img_True = Clean Image and Img_Test = Ouput of Network
# We want to have it so that PSNR2 > PSNR. This indicates that the PSNR of the PREDICTED image is greater than the PSNR of the ORIGINAL image

PSNR = peak_signal_noise_ratio(clean_testing_data[0], model_testing_data[0])
print("PSNR of Original Degraded Image: " + str(PSNR))
PSNR2 = peak_signal_noise_ratio(clean_testing_data[0], output_test[0])
print("PSNR of Predicted Clean Image: " + str(PSNR2))
print("\nIf the calculated difference is positive, our predicted image is of better quality than degraded")
print("If the calculated difference is negative, our predicted image is of worse quality than degraded")
print("\nDifference: " + str(PSNR2 - PSNR))

# Here, we compare the SSIM between (a) the GroundTruth and the NoisyImage. And (b) the GroundTruth and the PredictedImage
# Again, we aim to have SSIM2 > SSIM. This indicates that the SSIM of the PREDICTED image is greater than the SSIM of the ORIGINAL

SSIM = structural_similarity(clean_testing_data[0], model_testing_data[0])
print("SSIM of Original Degraded Image in Reference to Ground Truth: " + str(SSIM))
SSIM2 = structural_similarity(clean_testing_data[0], output_test[0])
print("SSIM of Predicted Clean Image in Reference to Ground Truth: " + str(SSIM2))
print("Difference: " + str(SSIM2 - SSIM))

# We can perform the same operation as above, but this time to one of the TRAINING images
# Recall, X_debug is a subset of training images with random noise (we defined this in the network itself)
output_train = model.predict(X_debug)

print("Train Image Sample...")

fig2, (ax1t, ax2t, ax3t) = plt.subplots(1, 3, sharey=True, figsize=(11, 11))
# Plot 1: Clean Testing Image
ax1t.imshow(z_debug[0], cmap="gray")
ax1t.title.set_text('Ground Truth')

# Plot 2: Degraded Testing Image
model_testing_data = np.array(model_testing_data)
ax2t.imshow(X_debug[0], cmap="gray")
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