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
from keras.callbacks import EarlyStopping

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

# The Keras API has two modes of constructing Neural Networks. The simplest is the Sequential Model which only allows for the layers
# to be added in sequence

'''
# Sequential Model
model = Sequential()

# Add an input layer which is similar to a feed_dict in TensorFlow
# Note that the input-shape must be a tuple containing the image-size
model.add(InputLayer(input_shape=(img_size_flat, )))

# The input is a flattened array with 784 elements
# but the convolutional layers expect images with shape (28, 28, 1)
model.add(Reshape(img_shape_full))

# First convolutional layer with Relu-activation and max-pooling
model.add(Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Second convolutional layer with Relu-activation and max-pooling
model.add(Conv2D(filters=36, kernel_size=5, strides=1, padding='same', activation='relu', name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Flatten the 4-rank output of the convolutional layers to 2-rank that can be input to a fully-connected  / dense layer
model.add(Flatten())

# First fully-connected / dense layer with Relu-activation
model.add(Dense(units=128, activation='relu'))

# Last fully-connected / dense layer with softmax-activation
# for use in classification
model.add(Dense(units=num_classes, activation='softmax'))

# Model Comlilation
optimizer = Adam(lr=1e-3)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(x=data.train.images, y=data.train.labels, epochs=1, batch_size=128)

# Evaluation
result = model.evaluate(x=data.test.images, y=data.test.labels)

print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))

# Prediction
images = data.test.images[0:9]
cls_true = data.test.cls[0:9]
y_pred = model.predict(x=images)
cls_pred = np.argmax(y_pred, axis=1)
plot_images(images=images, cls_true=cls_true, cls_pred=cls_pred)

y_pred = model.predict(x=data.test.images)
cls_pred = np.argmax(y_pred, axis=1)
plot_example_errors(cls_pred)

'''

# Functional Model
# The Keras API can also be used to construct more complicated networks using the Functional Model.
# This may look a little confusing at first, because each call to the Keras API will create and return an instance
# that is itself callable. It is not clear whether it is a function or an object - but we can call it as if it is a function.
# This allows us to build computational graphs that are more complex than the Sequential Model allows

# Create an input layer
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
model2 = Model(inputs=inputs, outputs=outputs)
model2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
# early-stopping
es = EarlyStopping(monitor='val_acc', patience=5)
model2.fit(x=data.train.images, y=data.train.labels, validation_split=0.1, epochs=1000, batch_size=128, callbacks=[es])

# Evaluation
result = model2.evaluate(x=data.test.images, y=data.test.labels)

print("{0}: {1:.2%}".format(model2.metrics_names[1], result[1]))

y_pred = model2.predict(x=data.test.images)
cls_pred = np.argmax(y_pred, axis=1)
plot_example_errors(cls_pred)

# save model
path_model = 'checkpoints/model.keras'
model2.save(path_model)