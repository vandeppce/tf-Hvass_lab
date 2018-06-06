import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

from keras.models import Sequential
from keras.layers import InputLayer, Input
from keras.layers import Reshape, MaxPooling2D
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam
from keras.models import Model, load_model

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

data.test.cls = np.argmax(data.test.labels, axis=1)

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
img_shape_full = (img_size, img_size, 1)
num_channels = 1
num_classes = 10

def plot_images(images, cls_true, cls_pred = None):
    assert len(images) == len(cls_true) == 9

    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace = 0.3, wspace = 0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def plot_example_errors(cls_pred):
    incorrect = (cls_pred != data.test.cls)

    images = data.test.images[incorrect]

    cls_pred = cls_pred[incorrect]

    cls_true = data.test.cls[incorrect]

    plot_images(images[0:9], cls_true[0:9], cls_pred[0:9])

inputs = Input(shape=(img_size_flat, ))

# Variable used for building the Nerual Network
net = inputs

# The input is an image as a flattened array with 784 elements
# But the convolutional layers expected images with shape (28, 28, 1
net = Reshape(img_shape_full)(net)

# First convolutional layer with relu-activation and max-pooling
net = Conv2D(filters=16, kernel_size=5, strides=1, padding='same', name='layer_conv1')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)

# Second convolutional layer with relu-activation and max-pooling
net = Conv2D(filters=36, kernel_size=5, strides=1, padding='same', name='layer_conv2')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)

# Flatten the output of the conv-layer from 4-dim to 2-dim
net = Flatten()(net)

# First fully-connected / dense layer with relu-activation
net = Dense(128, activation='relu')(net)

# Last fully-connected / dense layer with softmax-activation
net = Dense(units=num_classes, activation='softmax')(net)

outputs = net

# Model Compilation
# model2 = Model(inputs=inputs, outputs=outputs)
# model2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

num_networks = 3
num_iterasions = 1
batch_size = 64

'''
if True:
    for i in range(num_networks):
        print("Neural Network: {0}".format(i))
        model2 = Model(inputs=inputs, outputs=outputs)
        model2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        model2.fit(x=data.train.images, y=data.train.labels, epochs=num_iterasions, batch_size=batch_size)
        result = model2.evaluate(x=data.validation.images, y=data.validation.labels, batch_size=64)
        print(result[1])
        model2.save(filepath='checkpoints/keras_ensemble/networks_' + str(i))
        print()
'''

def predict_labels(model, images):
    return model.predict(x=images, batch_size=64)

def ensemble_prediction():
    pred_labels = []
    test_accs = []

    for i in range(num_networks):
        path = 'checkpoints/keras_ensemble/networks_' + str(i)
        model = load_model(filepath=path)

        pred = predict_labels(model, data.test.images)
        pred_cls = np.argmax(pred, axis=1)

        correct = (pred_cls == data.test.cls)
        acc = sum(correct) / len(correct)

        test_accs.append(acc)
        pred_labels.append(pred)

    return np.array(pred_labels), np.array(test_accs)

pred_labels, test_accs = ensemble_prediction()
print(test_accs)
# print(pred_labels.shape)              # (3, 10000, 10)
ensemble_pred = np.mean(pred_labels, axis=0)
ensemble_cls = np.argmax(ensemble_pred, axis=1)

ensemble_acc = sum((ensemble_cls == data.test.cls)) / len(ensemble_cls)
print(ensemble_acc)