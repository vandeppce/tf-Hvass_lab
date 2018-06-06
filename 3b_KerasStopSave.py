import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

from keras.models import Sequential
from keras.layers import InputLayer, Input
from keras.layers import Reshape, MaxPooling2D
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam
from keras.models import Model
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

# Input
inputs = Input(shape=(img_size_flat, ))

# Reshape
outputs = Reshape(img_shape_full)(inputs)

# Conv_1 and Max_1
outputs = Conv2D(filters=16, kernel_size=5, strides=1, padding='same', name='layer_conv1')(outputs)
outputs = MaxPooling2D(pool_size=2, strides=2)(outputs)

# Conv_2 and Max_2
outputs = Conv2D(filters=36, kernel_size=5, strides=1, padding='same', name='layer_conv2')(outputs)
outputs = MaxPooling2D(pool_size=2, strides=2)(outputs)

# flatten
outputs = Flatten()(outputs)

# fc_1
outputs = Dense(units=128, activation='relu', name='fc1')(outputs)
# fc_2
outputs = Dense(units=num_classes, activation='softmax', name='fc2')(outputs)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

'''
# Early-stopping
es = EarlyStopping(monitor='val_acc', patience=5)

# training
model.fit(x=data.train.images, y=data.train.labels, batch_size=128, epochs=200, callbacks=[es], validation_split=0.1)

# evaluation
result = model.evaluate(x=data.validation.images, y=data.validation.labels)
print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))

# predict
y_pred = model.predict(x=data.test.images)
cls_pred = np.argmax(y_pred, axis=1)

model.save('checkpoints/model.keras')
'''
# another implement
def run(num_iterations, model):
    best_validation_accuracy = 0.0
    last_improvement = 0
    require_improvement = 6
    total_iterations = 0

    for i in range(num_iterations):
        total_iterations += 1

        print("epoch: {0}".format(str(total_iterations)))

        model.fit(x=data.train.images, y=data.train.labels, batch_size=128, epochs=1)

        if total_iterations % 2 == 0:
            result = model.evaluate(x=data.validation.images, y=data.validation.labels)
            acc_validation = result[1]

            if acc_validation > best_validation_accuracy:
                best_validation_accuracy = acc_validation
                last_improvement = total_iterations
                model.save('checkpoints/keras.model.earlystop')

                print("* Validation acc: {0}".format(acc_validation))

            else:
                print("- Validation acc: {0}".format(acc_validation))

        if total_iterations - last_improvement > require_improvement:
            print("No improvement found in a while")
            break
    return model

model = run(200, model)
result = model.evaluate(x=data.validation.images, y=data.validation.labels)
print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))