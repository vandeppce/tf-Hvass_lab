#
# Transfer Learning: Re-route the output of the original model just
# prior to its classification layers and instead use a new classifier
# that we had created.

# Because the original model was 'frozen' its weights could not be further
# optimized, so whatever had been learned by all the previous layers in the
# model, could not be fine-tuned to the new dataset.

# In this work, we use VGG-16. The dense layers are responsible for combining
# features from the convolutional layers and this helps in the final classification.
# So when the VGG-16 model is used on another dataset we may have to replace
# all the dense layers. In this case we add another dense-layer and a
# dropout-layer to  avoid overfitting.

# The difference between Transfer Learning and Find-Tuning is that in Transfer
# Learning we only optimized the weights of the new classification layers we
# have added, while we keep the weights of the original VGG16 model. In Fine-Tuning
# we optimize both the weights of the new classification layers we have added,
# as well as some or all of the layers from the VGG-16 model.

import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
import os

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop

from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import knifey


# Helper-function for joining a directory and list of filenames
def path_join(dirname, filenames):
    return [os.path.join(dirname, filename) for filename in filenames]


# Helper-function for plotting images
def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true)

    fig, axes = plt.subplots(3, 3)

    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], interpolation=interpolation)
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

# Helper-function for loading images
def load_images(image_paths):
    images = [plt.imread(path) for path in image_paths]

    return np.asarray(images)


# Helper-function for plotting training history
def plot_training_history(history):
    # Get the classification accuracy and loss-value
    # for the training-set.
    acc = history.history['categorical_accuracy']
    loss = history.history['loss']

    # Get it for the validation-set (we only use the test-set).
    val_acc = history.history['val_categorical_accuracy']
    val_loss = history.history['val_loss']

    # Plot the accuracy and loss-values for the training-set.
    plt.plot(acc, linestyle='-', color='b', label='Training Acc.')
    plt.plot(loss, 'o', color='b', label='Training Loss')

    # Plot it for the test-set.
    plt.plot(val_acc, linestyle='--', color='r', label='Test Acc.')
    plt.plot(val_loss, 'o', color='r', label='Test Loss')

    # Plot title and legend.
    plt.title('Training and Test Accuracy')
    plt.legend()

    # Ensure the plot shows correctly.
    plt.show()


# Dataset
knifey.maybe_download_and_extract()
knifey.copy_files()

train_dir = knifey.train_dir
test_dir = knifey.test_dir

# Pre-Trained Model: VGG-16
# The VGG16 model contains a convolutional part and a fully-connected
# (or dense) part which is used for classification. If include_top=True
# then the whole VGG16 model is downloaded which is about 528MB. If
# include_top=False then only the convolutional part of the VGG16 model
# is downloaded which is just 57 MB

model = VGG16(include_top=True, weights='imagenet')

input_shape = model.layers[0].output_shape[1:3]  # (None, 224, 224, 3)

dategen_train = ImageDataGenerator(
    rescale=1. / 255,  # 重缩放因子
    rotation_range=180,  # 数据提升时图片随机转动的角度
    width_shift_range=0.1,  # 数据提升时图片水平偏移的角度
    height_shift_range=0.1,  # 数据提升时图片垂直偏移的角度
    shear_range=0.1,  # 剪切强度
    zoom_range=[0.9, 1.5],  # 随机缩放的幅度
    horizontal_flip=True,  # 随机水平翻转
    vertical_flip=True,  # 随机垂直翻转
    fill_mode='nearest'  # 边界点处理
)

dategen_test = ImageDataGenerator(
    rescale=1. / 255
)

# The data-generators will return batches of images. Because the VGG16 model is so large,
# the batch-size cannot be too large, otherwise you will run out of RAM on the GPU.

batch_size = 20

if True:
    save_to_dir = None
else:
    save_to_dir = 'augmented_images/'

# Now we create the actual data-generator that will read files from disk,
# resize the images and return a random batch.

generator_train = dategen_train.flow_from_directory(directory=train_dir,
                                                    target_size=input_shape,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    save_to_dir=save_to_dir)

generator_test = dategen_test.flow_from_directory(directory=test_dir,
                                                  target_size=input_shape,
                                                  batch_size=batch_size,
                                                  shuffle=False)

# Because the data-generators will loop for eternity, we need to specify the number of
# steps to perform during evaluation and prediction on the test-set. Because our test-set
# contains 530 images and the batch-size is set to 20, the number of steps is 26.5 for one
# full processing of the test-set. This is why we need to reset the data-generator's
# counter in the example_errors() function above, so it always starts processing from the
# beginning of the test-set.

steps_test = generator_test.n / batch_size

image_paths_train = path_join(train_dir, generator_train.filenames)
image_paths_test = path_join(test_dir, generator_test.filenames)

cls_train = generator_train.classes
cls_test = generator_test.classes
cls_true = cls_test
class_names = list(generator_train.class_indices.keys())
num_classes = generator_train.num_classes

# Class Weights
# The Knifey-Spoony dataset is quite imbalanced because it has few images of forks, and many
# images of spoons. This can cause a problem during training because the neural network will
# be shown many more examples of spoons than forks, so it might become bettor at recognizing
# spoons. Here we use sklearn to calculate weights that will properly balance the dataset.
# These weights are applied to the gradient for each image in the batch during training, so
# as to scale their influcnce on the overall gradient for the batch.

class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(cls_train), y=cls_train)

print(class_weight)
print(class_names)


# Helper-function for printing confusion
def print_confusion_matrix(cls_pred):
    cm = confusion_matrix(y_true=cls_test, y_pred=cls_pred)
    print("Confusion matrix: ")

    print(cm)

    for i, class_name in enumerate(class_names):
        print("({0}) {1}".format(i, class_name))


# Helper-function for plotting example errors
def plot_example_errors(cls_pred, cls_true):
    incorrect = (cls_pred != cls_test)

    image_paths = np.array(image_paths_test)[incorrect]
    images = load_images(image_paths=image_paths[0:9])

    cls_pred = cls_pred[incorrect]
    cls_true = cls_true[incorrect]

    plot_images(images=images, cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])


def example_errors():
    # The  Keras data-generator for the test-set must be reset
    # before processing. This is because the generator will loop
    # infinitely and keep an internal index into the dataset.
    # So it might start in the middle of the test-set if we do
    # not reset it first. This makes it impossible to match the
    # predicted classes with the input images.
    # If we reset the generator, then it always starts at the
    # beginning so we know exactly which input-images were used.

    generator_test.reset()
    y_pred = new_model.predict_generator(generator=generator_test, steps=steps_test)

    cls_pred = np.argmax(y_pred, axis=1)
    plot_example_errors(cls_pred, cls_true)
    print_confusion_matrix(cls_pred)

# Example Predictions
# Using pre-trained VGG16 model for prediction.

def predict(image_path):
    img = PIL.Image.open(image_path)
    img_resized = img.resize(input_shape, PIL.Image.LANCZOS)

    plt.imshow(img_resized)
    plt.show()

    img_array = np.expand_dims(np.array(img_resized), axis=0)

    # Use the VGG16 model to make a prediction.
    # This outputs an array with 1000 numbers corresponding to
    # the classes of the ImageNet-dataset.

    pred = model.predict(img_array)

    # Decode the output of the VGG16 model.
    pred_decoded = decode_predictions(pred)[0]

    for code, name, score in pred_decoded:
        print("{0:>6.2%} : {1}".format(score, name))


# predict(image_path='images/parrot_cropped1.jpg')
# predict(image_path=image_paths_train[0])
# predict(image_path=image_paths_train[1])
# predict(image_paths_test[0])

# Transfer Learning
# The pre-trained VGG16 model was unable to classify images from the knifey-spoony dataset.
# The reason is perhaps that the VGG16 model was trained on the so-called ImageNet dataset
# which may not have many images of cutlery.

# The lower layers of a Convolutional Neural Network can recognize many different shapes of
# features in an image. It is the last few fully-connected layers that combine these features
# into classification of a whole image. So we can try and re-route the output of the last
# convolutional layer of the VGG16 model to a new fully-connected neural network that we
# create for doing classification on the Knifey-Spoony dataset.

model.summary()
transfer_layer = model.get_layer('block5_pool')
# 查看输出
# print(transfer_layer.output)

# 建立模型
# 特征模型（迁移）

conv_model = Model(inputs=model.input, outputs=transfer_layer.output)

# 分类模型
new_model = Sequential()
new_model.add(conv_model)
new_model.add(Flatten())
new_model.add(Dense(1024, activation='relu'))
new_model.add(Dropout(0.5))
new_model.add(Dense(num_classes, activation='softmax'))

'''
inputs = model.input
net = transfer_layer.output
net = Flatten()(net)
net = Dense(1024, activation='relu')(net)
net = Dropout(0.5)(net)
net = Dense(num_classes, activation='softmax')(net)
outputs = net
'''
optimizer = Adam(lr=1e-5)
loss = 'categorical_crossentropy'
metrics = ['categorical_accuracy']


# model2 = Model(inputs=inputs, outputs=outputs)
# model2.compile(optimizer, loss, metrics)
# model2.summary()

def print_layer_trainable():
    for layer in conv_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))


print_layer_trainable()

# In Transfer Learning we are initially only interested in reusing the pretrained VGG16 model
# as it is, so we will disable training for all its layers.

conv_model.trainable = False
for layer in conv_model.layers:
    layer.trainable = False

print_layer_trainable()

new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# An epoch normally means one full processing of the training-set.
# But the data-generator that we created above, will produce batches of training-data for eternity.
# So we need to define the number of steps we want to run for each "epoch"
# and this number gets multiplied by the batch-size defined above.
# In this case we have 100 steps per epoch and a batch-size of 20,
# so the "epoch" consists of 2000 random images from the training-set.
# We run 20 such "epochs".

epochs = 20
steps_per_epoch = 100

history = new_model.fit_generator(generator=generator_train,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)

# Keras records the performance metrics at the end of each "epoch" so they can be plotted later.
# This shows that the loss-value for the training-set generally decreased during training,
# but the loss-values for the test-set were a bit more erratic. Similarly,
# the classification accuracy generally improved on the training-set
# while it was a bit more erratic on the test-set.

plot_training_history(history)

# evaluate
result = new_model.evaluate_generator(generator=generator_test, steps=steps_test)
print("Test-set classification accuracy: {0:.2%}".format(result[1]))

# plot example errors
example_errors()

# Fine-Tuning
# In Transfer Learning the original pre-trained model is locked or frozen during training of the new classifier.
# This ensures that the weights of the original VGG16 model will not change.
# One advantage of this, is that the training of the new classifier will not propagate large gradients
# back through the VGG16 model that may either distort its weights or cause overfitting to the new dataset.

# But once the new classifier has been trained we can try and gently fine-tune
# some of the deeper layers in the VGG16 model as well. We call this Fine-Tuning.
# It is a bit unclear whether Keras uses the trainable boolean in each layer of the original VGG16 model
# or if it is overrided by the trainable boolean in the "meta-layer" we call conv_layer.
# So we will enable the trainable boolean for both conv_layer and all the relevant layers in the original VGG16 model.

conv_model.trainable = True

for layer in conv_model.layers:
    # Boolean whether this layer is trainable.
    trainable = ('block5' in layer.name or 'block4' in layer.name)

    # Set the layer's bool.
    layer.trainable = trainable

print_layer_trainable()

# We will use a lower learning-rate for the fine-tuning
# so the weights of the original VGG16 model only get changed slowly.

optimizer_fine = Adam(lr=1e-7)
new_model.compile(optimizer=optimizer_fine, loss=loss, metrics=metrics)
history = new_model.fit_generator(generator=generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)

plot_training_history(history)
result = new_model.evaluate_generator(generator_test, steps=steps_test)
print("Test-set classification accuracy: {0:.2%}".format(result[1]))
example_errors()