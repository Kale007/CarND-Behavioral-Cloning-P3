import os
import csv
import cv2
import sklearn
import matplotlib

import matplotlib.pyplot as plt
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

### Define Hyperparameters ###
# model params
epochs = 7
batch_size = 128
validation_split = 0.2
size = 8000

# systems params
num_cameras = 3
steering_correction_left_cam = 0.3
steering_correction_right_cam = -0.2


### Load data ###
# Extract lines from images
lines = []
with open('data/AllTrainingData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    is_first = True
    for line in reader:
        # skip first line (data titles)
        if is_first == True:
            is_first = False
            continue
        lines.append(line)

corrections_c_l_r = [0, steering_correction_left_cam, steering_correction_right_cam]

def input_ouput_gen(sample_set):
    images = []
    measurements = []
    for timepoint in range(len(sample_set)):
        for i in range(num_cameras):
            # generalize filename path and append to images list
            timepoint_data = sample_set[timepoint]
            source_path = timepoint_data[i]
            filename = source_path.split('/')[-1]
            current_path = 'data/AllTrainingData/IMG/' + filename
            image = cv2.imread(current_path)

            steering_center = float(timepoint_data[3]) # center steering measurement
            steering_measurement = steering_center + corrections_c_l_r[i]

            images.append(image)
            measurements.append(steering_measurement)

    return images, measurements

# define functions to sample evenly distributed steering inputs and images
###### TAKEN FROM Github user manavkataria #########
def bin_dataset(y_train):
    counts, bin_edges = np.histogram(y_train, bins='auto')
    bin_ids = np.digitize(y_train, bin_edges)
    nbins = len(bin_edges)

    # Initialize Bins
    bins = [[] for _ in range(nbins)]
    # Build Reverse Index
    for i, y in enumerate(y_train):
        rev_idx = bin_ids[i] - 1
        bins[rev_idx].append(i)

    return bins, counts, bin_edges


def construct_balanced_dataset_from_bins(X_train, y_train, bins, size):
    X_balanced = np.empty((size, 160, 320, 3), dtype=np.float32)
    y_balanced = np.empty((size,), dtype=np.float32)

    for i, idx in enumerate(random_uniform_sampling_from_bins(bins)):
        X_balanced[i, :, :, :] = X_train[idx]
        y_balanced[i] = y_train[idx]
        if i >= (size - 1):
            break

    return X_balanced, y_balanced


def random_uniform_sampling_from_bins(bins):
    while 1:
        bin_id = np.random.randint(len(bins))
        selected_bin_indices = bins[bin_id]
        bin_length = len(selected_bin_indices)
        if bin_length <= 0:
            # Pick another bin; this one being empty
            continue
        rev_idx = np.random.randint(bin_length)
        yield selected_bin_indices[rev_idx]


def balance_dataset(X_train, y_train, size=size):
    bins, _, bin_edges = bin_dataset(y_train)
    X_balanced, y_balanced = construct_balanced_dataset_from_bins(X_train, y_train, bins, size=size)
    return X_balanced, y_balanced
###### end TAKEN FROM Github user manavkataria #########



shuffle(lines)
input_images, output_steering = input_ouput_gen(lines)
X_balanced, y_balanced = balance_dataset(input_images, output_steering)

train_samples_per_epoch = int(len(X_balanced) * (1 - validation_split))
validation_samples_per_epoch = int(len(X_balanced) * validation_split)
print('Datapoints:' ,len(lines) * num_cameras)
print('Num of training samples: ', train_samples_per_epoch)
print('Num of validation samples: ', validation_samples_per_epoch)

# Import Keras methods
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Flatten, Lambda, Convolution2D, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()
print('Data SHAPE :', np.shape(X_balanced))
train_generator = train_datagen.flow(X_balanced, y_balanced, batch_size=batch_size)
validation_generator = validation_datagen.flow(X_balanced, y_balanced, batch_size=batch_size)


### Define neural network architecture ###
model = Sequential()

### Preprocess input
# normalize each image and mean centered
model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160, 320, 3)))
# Cropping layer to remove unwanted image sections
model.add(Cropping2D(cropping=((70, 25), (0, 0,)))) # input_shape=(160, 320, 3)))


## Model Layers ##

# # model.add(Conv2D(1, 1, 1, subsample=(1, 1), activation='elu'))
# model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
# model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
# model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
# model.add(Conv2D(64, 3, 3, subsample=(1, 1), activation='relu'))
# model.add(Conv2D(64, 3, 3, subsample=(1, 1), activation='relu'))
# model.add(Flatten())
# model.add(Dense(1164))
# model.add(Dropout(.1))
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(.1))
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(.1))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1))

# # Model C2
# model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='elu'))
# # model.add(Conv2D(1, 1, 1, subsample=(1, 1), border_mode="same"))
# # model.add(Activation('elu'))
# model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='elu'))
# model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='elu'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, 3, 3, activation='elu'))
# model.add(Conv2D(64, 3, 3, activation='elu'))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(100))
# model.add(Activation('elu'))
# model.add(Dense(50))
# model.add(Dropout(0.5))
# model.add(Activation('elu'))
# model.add(Dense(10))
# model.add(Activation('elu'))
# model.add(Dense(1))



# Model C - NVIDIA neural network described in Zieba et al.(2016)
# model.add(Conv2D(24, 8, 8, subsample=(2, 2), activation='relu'))
# model.add(Conv2D(36, 5, 5, subsample=(2, 2)))
# model.add(Dropout(0.5))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Activation('relu'))
# model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(64, 3, 3, activation='relu'))
# model.add(Conv2D(64, 3, 3))
# model.add(Dropout(0.2))
# model.add(Activation('relu'))
# model.add(Flatten())
# model.add(Dense(100))
# model.add(Activation('relu'))
# model.add(Dense(50))
# model.add(Activation('relu'))
# model.add(Dense(10))
# model.add(Dropout(0.5))
# model.add(Activation('relu'))
# model.add(Dense(1))

# # Model B - NVIDIA neural network described in Zieba et al.(2016)
model.add(Conv2D(1, 1, 1, subsample=(1, 1), activation='relu'))
model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1))

# Model Ab
# # convolutional layer with relu - filters=6, kernal=(5, 5), strides=(2, 2)
# model.add(Convolution2D(6, 5, 5, activation='relu'))
# # max-pooling layer
# model.add(MaxPooling2D())
# # convolutional layer with relu - filters=6, kernal=(5, 5), strides=(2, 2)
# model.add(Convolution2D(6, 5, 5, activation='relu'))
# # max-pooling layer
# model.add(MaxPooling2D())
# # flatten layer
# model.add(Flatten())
# # fully connected layers
# model.add(Dense(128))
# model.add(Dense(84))
# model.add(Dense(1))

### Training Pipeline ###
# Compile model with mean-squared-error and adam-optimizer
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=train_samples_per_epoch, validation_data=validation_generator, nb_val_samples=validation_samples_per_epoch, nb_epoch=epochs, verbose=1)

# Save model to local
model.save('model.h5')

# Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
