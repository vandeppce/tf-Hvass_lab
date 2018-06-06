import matplotlib.pyplot as plt
from matplotlib.image import imread

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os

import keras
from keras.layers import Input, Reshape, Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, BatchNormalization, Lambda
from keras.models import Model, load_model

from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.preprocessing import image as pi

import inception

import knifey
from knifey import num_classes

data_dir = knifey.data_dir
dataset = knifey.load()
class_names = dataset.class_names

image_paths_train, cls_train, labels_train = dataset.get_training_set()
image_paths_test, cls_test, labels_test = dataset.get_test_set()

# load images
def load_images(image_paths):
    images = [imread(image) for image in image_paths]

    return np.asarray(images)

# inception 模型
model = inception.Inception()

file_path_cache_train = os.path.join(data_dir, 'inception-knifey-train.pkl')
file_path_cache_test = os.path.join(data_dir, 'inception-knifey-test.pkl')

# 缓存
print("Processing Inception transfer-values for training-images ...")

# If transfer-values have already been calculated then reload them,
# otherwise calculate them and save them to a cache-file.
transfer_values_train = inception.transfer_values_cache(cache_path=file_path_cache_train,
                                              image_paths=image_paths_train,
                                              model=model)

print("Processing Inception transfer-values for test-images ...")
# If transfer-values have already been calculated then reload them,
# otherwise calculate them and save them to a cache-file.
transfer_values_test = inception.transfer_values_cache(cache_path=file_path_cache_test,
                                             image_paths=image_paths_test,
                                             model=model)

# keras模型
transfer_len = model.transfer_len

inputs = Input(shape=(int(transfer_len), ))
net = Dense(units=1024, activation='relu')(inputs)
net = Dense(units=4, activation='softmax')(net)

outputs = net

keras_model = Model(inputs=inputs, outputs=outputs)
keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
keras_model.fit(x=transfer_values_train, y=labels_train, batch_size=64)