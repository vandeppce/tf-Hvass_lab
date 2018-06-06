import matplotlib.pyplot as plt
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

import cifar10
from cifar10 import img_size, num_channels, num_classes

cifar10.maybe_download_and_extract()
class_names = cifar10.load_class_names()
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

model = inception.Inception()
file_path_cache_train = os.path.join(cifar10.data_path, 'inception_cifar10_train.pkl')
file_path_cache_test = os.path.join(cifar10.data_path, 'inception_cifar10_test.pkl')

print("Processing Inception transfer-values for training-images ...")

images_scaled = images_train * 255.0
transfer_values_train = inception.transfer_values_cache(cache_path=file_path_cache_train, images=images_scaled, model=model)

print("Processing Inception transfer-values for testing-images ...")

images_scaled = images_test * 255.0
transfer_values_test = inception.transfer_values_cache(cache_path=file_path_cache_test, images=images_scaled, model=model)

transfer_len = model.transfer_len

# transfer_values_train = np.reshape(transfer_values_train, (images_train.shape[0], transfer_len, 1))
# transfer_values_test = np.reshape(transfer_values_test, (images_test.shape[0], transfer_len, 1))

# shape的输入必须是一个数或者字符串，不能是变量
inputs = Input(shape=(int(transfer_len),))

net = Dense(1024, activation='relu')(inputs)
net = Dense(num_classes, activation='softmax')(net)

'''
inputs = Input(shape=[32, 64])

net = inputs
net = Flatten()(net)
net = Dense(units=1024, activation='relu')(net)
net = Dense(units=num_classes, activation='softmax')(net)
'''
outputs = net

model_keras = Model(inputs=inputs, outputs=outputs)
model_keras.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_keras.summary()

model_keras.fit(x=transfer_values_train, y=labels_train, batch_size=64, epochs=10)

result = model_keras.predict(x=transfer_values_test)
pred_cls = np.argmax(result, axis=1)

correct = (pred_cls == cls_test)
accuracy = correct.mean()
print("acc: " + str(accuracy))
model.close()