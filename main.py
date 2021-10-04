import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Bidirectional, TimeDistributed, Conv2D
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from keras.callbacks import ModelCheckpoint

# If using the expanded 256x256 Dataset, set image_size = 256
# If using the CIFAR Dataset, set image_size = 32
noise_val = 0.25
image_size = 256
num_epochs = 100
datadir = "/home/zeshanfayyaz/LIA/Local_Images/Train/"
testdir = "/home/zeshanfayyaz/LIA/Local_Images/Test/"
test_output = "/home/zeshanfayyaz/LIA/NoiseRemoval/BETA/15GRU/Results/"  # Where we want to save our test images and metrics


def main():
    print("BETA: 12GRU")
    # Degrade Function which takes the clean image as input and returns the noisy image and the noise itself.
    # Same dimensionality, I.e. (256,256)
    def degrade(image, noise_val):
        nv = noise_val*255
        beta = nv * np.random.rand(1)
        image.astype('float32')

        g_col = np.random.normal(0, beta, image.shape[1])
        g_noise = np.tile(g_col, (image.shape[0], 1))
        g_noise = np.reshape(g_noise, image.shape)

        image_g = image + g_noise
        return image_g, g_noise

    # The Training + Validation Data are all the images in the datadir directory, resized to image_size, image_size
    # We shuffle the Data
    # Pass exception in case we are unable to read an image due to corruption - ignore that image
    def create_training_validation_data():
        training_validation_data = []
        for img in os.listdir(datadir):
            try:
                img_array = cv2.imread(os.path.join(datadir, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (image_size, image_size))
                new_array.astype('float32')
                training_validation_data.append([new_array, 0])
                random.shuffle(training_validation_data)
            except Exception as e:
                pass
        return training_validation_data

    # The test images are within the testdir, same dimensionality as the train images (256,256) or (32,32)
    # Pass exception in case we are unable to read an image due to corruption - ignore that image
    # We may also print the length of testing data to confirm correct file path
    def create_testing_data():
        testing_data = []
        for img in os.listdir(testdir):
            try:
                img_array = cv2.imread(os.path.join(testdir, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (image_size, image_size))
                new_array.astype('float32')
                testing_data.append([new_array])
            except Exception:
                pass
        print("Testing Data Length: " + str(len(testing_data)))
        testing_data = np.array(testing_data)
        return testing_data

    # We preprocess the testing_data created above and add a stripe noise with guassian value 'beta'
    # Normalize all images
    # We return lists containing the degraded test images, and clean test images, with equal lengths of testing_data
    def degrade_test_data():
        noisy_testing_data = []
        clean_testing_data = []
        for images in testing_data:
            degraded_image, _ = degrade(images[0], noise_val)
            noisy_testing_data.append(degraded_image.astype('float32') / 255.0)
            clean_testing_data.append(images.squeeze().astype('float32') / 255.0)
        clean_testing_data = np.array(clean_testing_data)
        noisy_testing_data = np.array(noisy_testing_data)
        return clean_testing_data, noisy_testing_data

    # We have previously read all images from datadir as training + validation images, here we perform the split
    # Default value for split is 85% training and 15% validation
    def training_validation_split(training_validation, split=0.85):
        print("Using " + str(split * 100) + "% " + "Train and " + str(100 - (split * 100)) + "% " "Validation")
        print("Total Training + Validation Length: " + str(len(training_validation)))
        numtosplit = int(split * (len(training_validation)))
        training_data = training_validation[:numtosplit]
        validation_data = training_validation[numtosplit:]
        print("Training Data Length: " + str(len(training_data)))
        print("Validation Data Length: " + str(len(validation_data)))
        return training_data, validation_data

    # Call this function when we want to inspect 1 image as: degraded, ground truth, and predicted
    # "type" argument refers to setting the title of the subplot as "Testing Image" or "Training Image"
    def create_subplots(degraded, ground_truth, predicted, type):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(11, 11))
        # Plot 1: Degraded Image
        ax1.imshow(degraded, cmap="gray")
        ax1.title.set_text('Degraded Image')

        # Plot 2: Ground Truth Image
        ax2.imshow(ground_truth, cmap="gray")
        ax2.title.set_text('Ground truth Image')

        # Plot 3: Predicted Clean Image
        ax3.imshow(predicted, cmap="gray")
        ax3.title.set_text('Predicted Image')
        if type == "test":
            fig.suptitle("Testing Image")
        elif type == "train":
            fig.suptitle("Training Image")
        return fig

    def average(lst):
        return sum(lst) / len(lst)

    # Calculate the PSNR and SSIM metrics using sklearn built-in calculations
    # All calculations are performed with respect to the ground_truth image
    # For both SSIM and PSNR, we aim for a large positive difference
    # If the difference is positive, our network is learning. Else, the predicted image is worse quality than degraded
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

    # MODEL 4: 15 BiGRU
    # Transpose the input shape replacing rows with columns and columns with rows
    # Return sequences is TRUE as we want an output for every timestep, and not a "many-to-one" output
    # Merge_mode is set to AVERAGE - in order to maintain dimensionality (256,256) [default is CONCAT]
    def train_model(image_size):
        inputs = Input(shape=(image_size, image_size))
        inputs_t = tf.transpose(inputs, perm=[0, 2, 1])
        output_1 = (Bidirectional(GRU(image_size, return_sequences=True), merge_mode="ave"))(inputs_t)
        # The input of layer 2 is the output of layer 0 [transpose] SUBTRACT the output of layer 1
        
        input_2 = tf.keras.layers.Subtract()([inputs_t, output_1])
        output_2 = (Bidirectional(GRU(image_size, return_sequences=True), merge_mode="ave"))(input_2)
        
        input_3 = keras.layers.Subtract()([inputs_t, output_2])
        output_3 = (Bidirectional(GRU(image_size, return_sequences=True), merge_mode="ave"))(input_3)
        
        input_4 = keras.layers.Subtract()([inputs_t, output_3])
        output_4 = (Bidirectional(GRU(image_size, return_sequences=True), merge_mode="ave"))(input_4)
        
        input_5 = keras.layers.Subtract()([inputs_t, output_4])
        output_5 = (Bidirectional(GRU(image_size, return_sequences=True), merge_mode="ave"))(input_5)
        
        input_6 = keras.layers.Subtract()([inputs_t, output_5])
        output_6 = Bidirectional(GRU(image_size, return_sequences=True), merge_mode="ave")(input_6)

        input_7 = keras.layers.Subtract()([inputs_t, output_6])
        output_7 = Bidirectional(GRU(image_size, return_sequences=True), merge_mode="ave")(input_7)

        input_8 = keras.layers.Subtract()([inputs_t, output_7])
        output_8 = Bidirectional(GRU(image_size, return_sequences=True), merge_mode="ave")(input_8)

        input_9 = keras.layers.Subtract()([inputs_t, output_8])
        output_9 = Bidirectional(GRU(image_size, return_sequences=True), merge_mode="ave")(input_9)
        
        input_10 = keras.layers.Subtract()([inputs_t, output_9])
        output_10 = (Bidirectional(GRU(image_size, return_sequences=True), merge_mode="ave"))(input_10)
        
        input_11 = keras.layers.Subtract()([inputs_t, output_10])
        output_11 = Bidirectional(GRU(image_size, return_sequences=True), merge_mode="ave")(input_11)

        input_12 = keras.layers.Subtract()([inputs_t, output_11])
        output_12 = Bidirectional(GRU(image_size, return_sequences=True), merge_mode="ave")(input_12)
	
	input_13 = keras.layers.Subtract()([inputs_t, output_12])
        output_13 = (Bidirectional(GRU(image_size, return_sequences=True), merge_mode="ave"))(input_13)
        
        input_14 = keras.layers.Subtract()([inputs_t, output_13])
        output_14 = Bidirectional(GRU(image_size, return_sequences=True), merge_mode="ave")(input_14)

        input_15 = keras.layers.Subtract()([inputs_t, output_14])
        output_15 = Bidirectional(GRU(image_size, return_sequences=True), merge_mode="ave")(input_15)

        # Perform TimeDistributed Operation to final output of GRU        
        output_GRU = TimeDistributed(Dense(image_size))(output_15)

        # Transpose the image once again, giving us original dimensionality
        output_GRU = tf.transpose(output_GRU, perm=[0, 2, 1])

        # We aim for the real_output to be as close as possible to 'z' -> the Clean Image
        # The clean image is the dirty image [Input] SUBTRACT the noise [Output of the Network (output_GRU)]
        real_output = tf.keras.layers.Subtract()([inputs, output_GRU])

        model = Model(inputs=inputs, outputs=real_output)
        model.summary()
        print("Model Compiled")
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-5)
        model.compile(
            optimizer=opt,
            loss='mean_squared_error',
            metrics=['mse']
        )
        return model

    # We separate the datadir images into Training Data and Validation Data
    print("Creating Training and Validation Data...")
    training_validation_data = create_training_validation_data()
    training_data, validation_data = training_validation_split(training_validation_data)
    # training_data, validation_data = training_validation_split(training_validation_data)

    # We can perform the same operation on Testing Data
    print("Creating Testing Data...")
    testing_data = create_testing_data()
    testing_data = np.array(testing_data)

    # Degrade testing data
    print("Degrading Testing Data...")
    clean_testing_data, noisy_testing_data = degrade_test_data()

    filepath = "/home/zeshanfayyaz/LIA/NoiseRemoval/BETA/15GRU/model_checkpoint.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callback_list = [checkpoint]

    loss_train = []
    loss_val = []
    X_debug = []
    z_debug = []

    model = train_model(image_size)
    for i in range(num_epochs):
        X = []
        z = []
        X_validation = []
        z_validation = []

        for images, label in training_data:
            noisy_image, _ = degrade(images, noise_val)
            # We append the training images to X and z
            X.append(noisy_image.astype('float32') / 255.0)  # X is the Dirty Training Images
            z.append(images.astype('float32') / 255.0)  # z is the Clean Training Images [Target]

        for images, label in validation_data:
            noisy_image, _ = degrade(images, noise_val)
            X_validation.append(noisy_image / 255.0)  # X_validation is the Dirty Validation Images
            z_validation.append(images / 255.0)  # z_validation is the Clean Validation Images [Target]

        print("Adding Noise Instances... Done")
        X_debug = X[0:2]
        z_debug = z[0:2]
        X = np.array(X)
        z = np.array(z)
        X_validation = np.array(X_validation)
        z_validation = np.array(z_validation)
        X_debug = np.array(X_debug)
        z_debug = np.array(z_debug)
        print("Reshaping Arrays... Done")

        # Calculate loss
        loss_metrics = model.fit(X, z,
                                 batch_size=50,
                                 epochs=1,
                                 validation_data=(X_validation, z_validation),
                                 callbacks=callback_list
                                 )
        # Append metrics to their respective lists
        loss_train.append(loss_metrics.history['loss'])
        loss_val.append(loss_metrics.history['val_loss'])
        print("Done Epoch: " + str(i + 1))

    model.save("/home/zeshanfayyaz/LIA/NoiseRemoval/BETA/15GRU/stripe_noise_model.h5")

    # Once training is complete, plot the Training Loss
    plt.plot(loss_train)
    plt.title("Training Loss")
    plt.grid()
    plt.savefig(test_output + "TrainingLoss.pdf")
    plt.show()

    # Once training is complete, plot the Validation Loss
    plt.plot(loss_val)
    plt.title("Validation Loss")
    plt.grid()
    plt.savefig(test_output + "ValidationLoss.pdf")
    plt.show()

    # Once training is complete, plot the Training and Validation Loss on the same axis
    # Define range of x axis
    epoch_list = []
    for i in range(1, num_epochs + 1):
        epoch_list.append(i)

    plt.plot(epoch_list, loss_val, label='val loss')
    plt.plot(epoch_list, loss_train, label='train loss')
    plt.title("Training Loss and Validation Loss")
    plt.legend(loc='center right')
    plt.grid()
    plt.savefig(test_output + "TrainingValidationLoss.pdf")
    plt.show()

    # Predict clean images using our degraded test images
    output_test = model.predict(noisy_testing_data)

    # Next, over all outputs of the test we wish to calculate the average of the PSNR and SSIM metrics
    # We append the respective PSNR and SSIM values to their lists, and calculate the average of each list
    # We take the average of ALL test images: len(output_test) = len(testing_data)
    # Display (print) these results on console
    psnr_degraded_lst = []
    psnr_predicted_lst = []
    psnr_difference_lst = []
    ssim_degraded_lst = []
    ssim_predicted_lst = []
    ssim_difference_lst = []

    for i in range(len(output_test)):
        psnr_degraded, psnr_predicted, psnr_difference, ssim_degraded, ssim_predicted, ssim_difference = \
            psnr_ssim_metrics(clean_testing_data[i], output_test[i], noisy_testing_data[i])
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

    # We print an example image of output_test so we can visually inspect any loss of stripe noise
    print("Test Image Sample...")
    test_image = create_subplots(noisy_testing_data[0], clean_testing_data[0], output_test[0], "test")
    test_image.show()

    # Save 15 sample test images (subplots of ground truth, degraded, and predicted) and corresponding metrics
    # These files are saved in test_output
    for i in range(15):
        test_image = create_subplots(noisy_testing_data[i], clean_testing_data[i], output_test[i], "test")
        test_image.savefig(test_output + str(i) + ".pdf")

        psnr_degraded, psnr_predicted, psnr_difference, ssim_degraded, ssim_predicted, ssim_difference = \
            psnr_ssim_metrics(clean_testing_data[i], output_test[i], noisy_testing_data[i])

        file = open(test_output + str(i) + "_metrics.txt", "w")
        file.write("PSNR of Degraded Image wrt Ground Truth: " + str(psnr_degraded)
                   + " \nPSNR of Predicted Image wrt Ground Truth: " + str(psnr_predicted)
                   + " \nPSNR Difference: " + str(psnr_difference)
                   + " \nSSIM of  Degraded Image wrt Ground Truth: " + str(ssim_degraded)
                   + " \nSSIM of Predicted Image wrt Ground Truth: " + str(ssim_predicted)
                   + " \nSSIM Difference: " + str(ssim_difference))
        file.close()

    # We can perform the same operation as above, but this time to one of the training images
    # Recall, X_debug is a subset of training images with random noise (we defined this in the network itself)
    output_train = model.predict(X_debug)
    print("Train Image Sample...")
    debug_image = create_subplots(X_debug[0], z_debug[0], output_train[0], "train")

    debug_image.show()
    _, _, psnr_difference, _, _, ssim_difference = psnr_ssim_metrics(z[0], output_train[0], X[0])
    print("Sample Train Image: PSNR Difference: " + str(psnr_difference))
    print("Sample Train Image: SSIM Difference: " + str(ssim_difference))

    print(clean_testing_data[0].dtype)
    print(output_test[0].dtype)
    print(noisy_testing_data[0].dtype)
    print(z[0].dtype)
    print(output_train[0].dtype)
    print(X[0].dtype)


if __name__ == "__main__":
    main()
