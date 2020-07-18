import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import ntpath
import random

datadir = 'track'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names = columns)
pd.set_option('display.max_colwidth', -1)

def path_leaf(path):
  head, tail = ntpath.split(path)
  return tail
data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)
data.head()

# Get variation of steering data over collected images.
# Steering angle = 0 values are more than left or right turns.
# Hence, this overexposure to steering angle = 0 is adjusted by removing random images of that class

remove_values = []
nr_intervals = 25
samples_per_intervals = 400

_, steering_interval = np.histogram(data['steering'], nr_intervals)
# print(hist)
# print('\n',bins)

for j in range(nr_intervals):
  steering_val = []
  for i in range(len(data['steering'])):
    if data['steering'][i] >= steering_interval[j] and data['steering'][i] <= steering_interval[j+1]:
      steering_val.append(i)
  steering_val = shuffle(steering_val)
  steering_val = steering_val[samples_per_intervals:]
  remove_values.extend(steering_val)

# Loading Data into Arrays:

def load_img_steering_data(datadir, df):
  img_path = []
  steering = []
  steering_correction = 0.25
  for i in range(len(data)):
    # Collect paths
    idx_data = data.iloc[i]
    center = idx_data[0]
    left = idx_data[1]
    right = idx_data[2]

    # center image append
    img_path.append(os.path.join(datadir, center.strip()))
    steering.append(float(idx_data[3]))

    # left image append
    img_path.append(os.path.join(datadir, left.strip()))
    steering.append(float(idx_data[3])+steering_correction)

    # right image append
    img_path.append(os.path.join(datadir, right.strip()))
    steering.append(float(idx_data[3])-steering_correction)

  img_paths = np.asarray(img_path)
  steerings = np.asarray(steering)
  return img_paths, steerings

image_paths, steerings = load_img_steering_data(datadir + '/IMG', data)

# Separating Training and Validation Data:

X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)
print('Training Samples: {}\nValid Samples: {}'.format(len(X_train), len(X_valid)))

# Preprocessing/Augmentation Definitions and Pipeline:

def zoom(image):
  zm = iaa.Affine(scale=(1, 1.3))
  img = zm.augment_image(image)
  return img

def pan(image):
  pn = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
  img = pn.augment_image(image)
  return img

def brightness_adjust(image):
    bt = iaa.Multiply((0.2, 1.2))
    img = bt.augment_image(image)
    return img

def flip_img(image, steering_val):
    img = cv2.flip(image,1)
    steering_val = -steering_val
    return img, steering_val

def augment_img(image, steering_val):
    image = mpimg.imread(image)
    if np.random.rand() < 0.5:
      image = pan(image)
    if np.random.rand() < 0.5:
      image = zoom(image)
    if np.random.rand() < 0.5:
      image = brightness_adjust(image)
    if np.random.rand() < 0.5:
      image, steering_val = flip_img(image, steering_val)

    return image, steering_val

# Pipeline:
def preprocess_pipeline(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

# Training Batch Generator

def training_generator(image_paths, steering_ang, batch_size):

  while True: # Always true for continuous running
    # Empty arrays to store batch process results
    img_batch = []
    steering_batch = []

    for i in range(batch_size): # Only process for given batch size
      # Select a random image
      idx = random.randint(0, len(image_paths) - 1)
      # Apply random augmentation process
      img, steering = augment_img(image_paths[idx], steering_ang[idx])
      # Apply preprocessing pipeling
      img = preprocess_pipeline(img)
      # Append final values to batch arrays
      img_batch.append(img)
      steering_batch.append(steering)

    yield (np.asarray(img_batch), np.asarray(steering_batch))

# x_train_gen, y_train_gen = next(training_generator(X_train, y_train, 1))

# Validation Batch Generator

def validation_generator(image_paths, steering_ang, batch_size):

  while True: # Always true for continuous running
    # Empty arrays to store batch process results
    img_batch = []
    steering_batch = []

    for i in range(batch_size):
      # Same pipeline as training except augmentation processes not applied as this is validation.
      # Preprocessing is applied.
      idx = random.randint(0, len(image_paths) - 1)
      img = mpimg.imread(image_paths[idx])
      steering = steering_ang[idx]
      img = preprocess_pipeline(img)
      img_batch.append(img)
      steering_batch.append(steering)

    yield (np.asarray(img_batch), np.asarray(steering_batch))
# x_valid_gen, y_valid_gen = next(validation_generator(X_valid, y_valid, 1))

# Define the NVIDIA Model

def nvidia_model():
  model = Sequential()
  model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(66, 200, 3), activation='elu'))

  model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))
  model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu'))

  model.add(Convolution2D(64, 3, 3, activation='elu'))
  model.add(Convolution2D(64, 3, 3, activation='elu'))

  model.add(Flatten())
  model.add(Dense(100, activation = 'elu'))
  model.add(Dense(50, activation = 'elu'))
  model.add(Dense(10, activation = 'elu'))
  model.add(Dense(1))

  model.compile(loss='mse', optimizer=Adam(lr=1e-3))
  return model

model = nvidia_model()
print(model.summary())

history = model.fit_generator(training_generator(X_train, y_train, 100),
                                  steps_per_epoch=300,
                                  epochs=10,
                                  validation_data=validation_generator(X_valid, y_valid, 100),
                                  validation_steps=200,
                                  verbose=1,
                                  shuffle = 1)

model.save('model.h5')
from google.colab import files
files.download('model.h5')
