import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import os
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Bidirectional, TimeDistributed
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

image_size = 256
num_epochs = 100
datadir = "/home/zeshanfayyaz/LIA/Local_Images/Train/"
testdir = "/home/zeshanfayyaz/LIA/Local_Images/Test/"
test_output = "/home/zeshanfayyaz/LIA/test_output/"

def main():
    # Degrade Function which takes the clean image as input and returns the noisy image and the noise itself.
    # Same dimensionality, I.e. (256,256)
    def degrade(image, noise_val):
        beta = noise_val * np.random.rand(1)
        image.astype('float32')
        np.random.seed(0)
        g_col = np.random.normal(0, beta, image.shape[1])
        g_noise = np.tile(g_col, (image.shape[0], 1))
        g_noise = np.reshape(g_noise, image.shape)

        image_g = image + g_noise
        return image_g, g_noise

    # The Training + Validation Data is all the images in the datadir directory, resized to image_size, image_size
    def create_training_data():
        training_validation_data = []
        for img in os.listdir(datadir):
            try:
                img_array = cv2.imread(os.path.join(datadir, img), cv2.IMREAD_GRAYSCALE)
                # Resizing the Images to image_size,image_size
                new_array = cv2.resize(img_array, (image_size, image_size))
                training_validation_data.append([new_array, 0])
                # Shuffling the data
                random.shuffle(training_data)
            # Pass exception in case we are unable to read an image due to corruption, ignore that image
            except Exception as e:
                pass
        return training_validation_data

    def create_testing_data():
        testing_data = []
        for img in os.listdir(testdir):
            try:
                img_array = cv2.imread(os.path.join(testdir, img), cv2.IMREAD_GRAYSCALE)
                # Resize the images in testdir to image_size, image_size
                new_array = cv2.resize(img_array, (image_size, image_size))
                testing_data.append([new_array])
            # Pass exception in case we are unable to read an image due to corruption, ignore that image
            except Exception:
                pass
        print("Testing Data Length: " + str(len(testing_data)))
        testing_data = np.array(testing_data)
        return testing_data

    def degrade_test_data():
        model_testing_data = []
        clean_testing_data = []
        for images in testing_data:
            # We degrade the Test images with a value of 25
            model_image = degrade(images[0], 25)[0]
            # We normalize all pixel values (max 255.0)
            model_testing_data.append(model_image / 255.0)
            clean_testing_data.append(images.squeeze() / 255.0)
        clean_testing_data = np.array(clean_testing_data)
        model_testing_data = np.array(model_testing_data)
        return clean_testing_data, model_testing_data

    def training_validation_split(training_validation, split=0.85):
        print("Using " + str(split * 100) + "% " + "Train and " + str(100 - (split * 100)) + "% " "Validation")
        print("Total Training + Validation Length: " + str(len(training_validation)))
        numtosplit = int(split * (len(training_validation)))
        training_data = training_validation[:numtosplit]
        validation_data = training_validation[numtosplit:]
        print("Training Data Length: " + str(len(training_data)))
        print("Validation Data Length: " + str(len(validation_data)))
        return training_data, validation_data

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

    def train_model(image_size):
        inputs = Input(shape=(image_size, image_size))
        # Transpose the input shape replacing rows with columns and columns with rows
        inputs_t = tf.transpose(inputs, perm=[0, 2, 1])
        # We perform this manually for ease of visualization, however can be done in a for loop
        # MODEL 3
        # Layer 1
        # Return sequences is TRUE as we want an output for every timestep, and not a "many-to-one" output
        # Merge_mode is set to AVERAGE - in order to maintain dimensionality (256,256) [default is CONCAT]
        output_1 = (Bidirectional(GRU(image_size, return_sequences=True), merge_mode="ave"))(inputs_t)
        # The input of layer 2 is the output of layer 0 [transpose] SUBTRACT the output of layer 1
        input_2 = tf.keras.layers.Subtract()([inputs_t, output_1])
        output_2 = (Bidirectional(GRU(image_size, return_sequences=True), merge_mode="ave"))(input_2)
        input_3 = keras.layers.Subtract()([input_2, output_2])
        output_3 = (Bidirectional(GRU(image_size, return_sequences=True), merge_mode="ave"))(input_3)
        input_4 = keras.layers.Subtract()([input_3, output_3])
        output_4 = (Bidirectional(GRU(image_size, return_sequences=True), merge_mode="ave"))(input_4)
        input_5 = keras.layers.Subtract()([input_4, output_4])
        output_5 = (Bidirectional(GRU(image_size, return_sequences=True), merge_mode="ave"))(input_5)
        input_6 = keras.layers.Subtract()([input_5, output_5])
        output_6 = Bidirectional(GRU(image_size, return_sequences=True), merge_mode="ave")(input_6)
        # Perform TimeDistributed Operation to final output of GRU
        output_GRU = TimeDistributed(Dense(image_size))(output_6)
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
        return model

        # We separate the datadir images into Training Data and Validation Data
        print("Creating Training and Validation Data...")
        training_validation_data = create_training_data()
        training_data, validation_data = training_validation_split(training_validation_data)

        # We can perform the same operation on Testing Data
        print("Creating Testing Data...")
        testing_data = create_testing_data()
        testing_data = np.array(testing_data)

        # Degrade testing data
        print("Degrading Testing Data...")
        clean_testing_data, model_testing_data = degrade_test_data()

        # Set the number of epochs to 100
        # In this case, X is our input and z is our target - as opposed to the conventional X and y
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
                noisy_image, _ = degrade(images, 25)
                # We append the training images to X and z
                X.append(noisy_image / 255.0)  # X is the Dirty Training Images
                z.append(images / 255.0)  # z is the Clean Training Images [Target]

            for images, label in validation_data:
                noisy_image, _ = degrade(images, 25)
                X_validation.append(noisy_image / 255.0)  # X_validation is the Dirty Validation Images
                z_validation.append(images / 255.0)  # z_validation is the Clean Validation Images [Target]

            print("Adding Noise Instances... Done")

            print("Reshaping Arrays..")
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
                                     batch_size=100,
                                     epochs=1,
                                     validation_data=(X_validation, z_validation)
                                     )
            # Append metrics to their respective lists
            loss_train.append(loss_metrics.history['loss'])
            loss_val.append(loss_metrics.history['val_loss'])
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
        for i in range(1, num_epochs + 1):
            epoch_list.append(i)

        plt.plot(epoch_list, loss_val, label='val loss')
        plt.plot(epoch_list, loss_train, label='train loss')
        plt.title("Training Loss and Validation Loss")
        plt.legend(loc='center right')
        plt.grid()
        plt.show()

        # Test out network using the CLEAN TEST Images, as we defined at the top of the code
        output_test = model.predict(model_testing_data)

        psnr_degraded_lst = []
        psnr_predicted_lst = []
        psnr_difference_lst = []
        ssim_degraded_lst = []
        ssim_predicted_lst = []
        ssim_difference_lst = []

        for i in range(2000):
            psnr_degraded, psnr_predicted, psnr_difference, ssim_degraded, ssim_predicted, ssim_difference = \
                psnr_ssim_metrics(clean_testing_data[i], output_test[i], model_testing_data[i])
            psnr_degraded_lst.append(psnr_degraded)
            psnr_predicted_lst.append(psnr_predicted)
            psnr_difference_lst.append(psnr_difference)
            ssim_degraded_lst.append(ssim_degraded)
            ssim_predicted_lst.append(ssim_predicted)
            ssim_difference_lst.append(ssim_difference)

        psnr_degraded_average = average(psnr_degraded_lst)
        print("The average PSNR of all Test Degraded Images wrt Ground Truth: " + str(psnr_degraded_average))
        psnr_predicted_average = average(psnr_predicted_lst)
        print("The average PSNR of all Test Predicted Images wrt Ground Truth: " + str(psnr_predicted_average))
        psnr_difference_average = average(psnr_difference_lst)
        print("The average PSNR Difference: " + str(psnr_difference_average))

        ssim_degraded_average = average(ssim_degraded_lst)
        print("The average SSIM of all Test Degraded Images wrt Ground Truth: " + str(ssim_degraded_average))
        ssim_predicted_average = average(ssim_predicted_lst)
        print("The average SSIM of all Test Predicted Images wrt Ground Truth: " + str(ssim_predicted_average))
        ssim_difference_average = average(ssim_difference_lst)
        print("The average SSIM Difference: " + str(ssim_difference_average))

        # We print an example image of output_test so we can visually inspect any loss of stripe noise
        print("Test Image Sample...")
        test_image = create_subplots(model_testing_data[0], clean_testing_data[0], output_test[0], "test")
        test_image.show()

        for i in range(15):
            test_image = create_subplots(model_testing_data[i], clean_testing_data[i], output_test[i], "test")
            test_image.savefig(test_output + str(i) + ".png")

            psnr_degraded, psnr_predicted, psnr_difference, ssim_degraded, ssim_predicted, ssim_difference = \
                psnr_ssim_metrics(clean_testing_data[i], output_test[i], model_testing_data[i])

            file = open(test_output + str(i) + "_metrics.txt", "w")
            file.write("PSNR of Degraded Image wrt Ground Truth: " + str(psnr_degraded)
                       + " \nPSNR of Predicted Image wrt Ground Truth: " + str(psnr_predicted)
                       + " \nPSNR Difference: " + str(psnr_difference)
                       + " \nSSIM of  Degraded Image wrt Ground Truth: " + str(ssim_degraded)
                       + " \nSSIM of Predicted Image wrt Ground Truth: " + str(ssim_predicted)
                       + " \nSSIM Difference: " + str(ssim_difference))
            file.close()

        # Comparing PSNR and SSIM
        # Compare PSNR where Img_True = Clean Testing Data and Img_Test = Degraded Original Testing Data
        # Then, compare PSNR2 where Img_True = Clean Image and Img_Test = Output of Network
        # PSNR2 > PSNR. This indicates that the PSNR of the PREDICTED image is greater than the PSNR of the ORIGINAL image

        # We can perform the same operation as above, but this time to one of the TRAINING images
        # Recall, X_debug is a subset of training images with random noise (we defined this in the network itself)
        output_train = model.predict(X_debug)
        print("Train Image Sample...")
        debug_image = create_subplots(X_debug[0], z_debug[0], output_train[0], "train")
        debug_image.show()
        _, _, psnr_difference, _, _, ssim_difference = psnr_ssim_metrics(z[0], output_train[0], X[0])
        print("Sample Train Image: PSNR Difference: " + str(psnr_difference))
        print("Sample Train Image: SSIM Difference: " + str(ssim_difference))


if __name__ == "__main__":
    main()

