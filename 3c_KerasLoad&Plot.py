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

import cifar10
from cifar10 import img_size, num_channels, num_classes

cifar10.maybe_download_and_extract()
class_names = cifar10.load_class_names()
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

img_size_cropped = 24


def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'
        ax.imshow(images[i, :, :, :],
                  interpolation=interpolation)
        cls_true_name = class_names[cls_true[i]]
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            cls_pred_name = class_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def pre_process_image(image, training):
    if training:
        image = tf.random_crop(value=image, size=[img_size_cropped, img_size_cropped, num_channels])
        image = tf.image.random_flip_left_right(image)

        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper = 1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)

        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)

    else:
        image = tf.image.resize_image_with_crop_or_pad(image, target_height=img_size_cropped, target_width=img_size_cropped)

    return image

def pre_process(images, training):

    images = tf.map_fn(fn=lambda image: pre_process_image(image, training=True), elems=images)
    return images

def plot_conv_weights(weights, input_channel=0):
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(weights)
    w_max = np.max(weights)

    # Number of filters used in the conv. layer.
    num_filters = weights.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = int(math.ceil(math.sqrt(num_filters)))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = weights[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def plot_conv_output(values):
    num_filters = values.shape[3]

    num_grids = int(math.ceil(math.sqrt(num_filters)))

    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = values[0, :, :, i]

            ax.imshow(img, interpolation='nearest', cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

# Keras
inputs = Input(shape=[img_size, img_size, num_channels], name='x')

# keras tensor 和tensorflow tensor的结构略有不同
# tf每次操作读的是整个batch，而keras则传入batch中的某一个样本
# 因此直接用Model对tf的tensor操作则会导致keras整体model的混乱
# 因此需要对操作的tf tensor加一个Lambda层
# Lambda层中是tf对keras tensor的操作
# 经过Lambda层tf tensor就变成了keras tensor

distorted_images = Lambda(lambda x: pre_process(x, training=True), name='pre_process')(inputs)
net = distorted_images
# net = inputs

net = Conv2D(filters=64, kernel_size=5, padding='same')(net)
net = BatchNormalization(axis=3, name='layer_conv1')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)

net = Conv2D(filters=64, kernel_size=5, padding='same', name='layer_conv2')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)

net = Flatten()(net)
net = Dense(units=256, activation='relu')(net)
net = Dense(units=128, activation='relu')(net)
net = Dense(units=num_classes, activation='softmax')(net)

outputs = net
'''
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
es = EarlyStopping(monitor='val_acc', patience=5)
model.fit(x=images_train, y=labels_train, batch_size=64, validation_split=0.1, callbacks=[es], epochs=1)
model.save(filepath='checkpoints/cifar_keras/cifar.keras')
'''

# keras 直接load_model经常出错
# 这里建立一个和源模型一样的模型，然后load_weights()
model_load = Model(inputs=inputs, outputs=outputs)
model_load.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_load.load_weights('checkpoints/cifar_keras/cifar.keras')

model_load.summary()

pred_label = model_load.predict(x=images_test, batch_size=64)
pred_cls = np.argmax(pred_label, axis=1)
correct = (pred_cls == cls_test)
acc = np.mean(correct)
print("acc: {0}".format(acc))

layer_input = model_load.layers[0]
layer_conv1 = model_load.layers[2]
layer_conv_bn = model_load.layers[3]
layer_conv2 = model_load.layers[5]

weights_conv1 = layer_conv1.get_weights()[0]
weights_conv2 = layer_conv2.get_weights()[0]

def get_test_image(i):
    return images_test[i, :, :, :], cls_test[i]

img, cls = get_test_image(20)
img = np.reshape(img, (1, img_size, img_size, num_channels))
# 绘制权重
# plot_conv_weights(weights_conv1, input_channel=1)
# plot_conv_weights(weights_conv2, input_channel=1)

layer_out1 = Model(inputs=layer_input.input, outputs=layer_conv1.output)
layer_out2 = Model(inputs=layer_input.input, outputs=layer_conv2.output)

out1 = layer_out1.predict(x = img)
out2 = layer_out2.predict(x = img)

# 绘制输出
plot_conv_output(out1)
plot_conv_output(out2)