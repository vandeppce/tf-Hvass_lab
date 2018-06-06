import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
import prettytensor as pt

import cifar10
from cifar10 import img_size, num_channels, num_classes

# 保存路径
cifar10.data_path = "data/CIFAR-10/"

# 如果没有找到则自动下载
cifar10.maybe_download_and_extract()

# 载入分类名称
class_names = cifar10.load_class_names()
# print(class_names)

images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

# print("Size of:")
# print("- Training-set:\t\t{}".format(len(images_train)))
# print("- Test-set:\t\t{}".format(len(images_test)))

# print(images_train.shape, cls_train.shape, labels_train.shape)
img_size_cropped = 24

def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true) == 9

    fig, axes = plt.subplots(3, 3)

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


# Get the first images from the test-set.
# images = images_test[0:9]

# Get the true classes for those images.
# cls_true = cls_test[0:9]

# Plot the images and labels using our helper-function above.
# plot_images(images=images, cls_true=cls_true, smooth=False)
# plot_images(images=images, cls_true=cls_true, smooth=True)

x = tf.placeholder(dtype=tf.float32, shape=(None, img_size, img_size, num_channels), name='x')
y_true = tf.placeholder(dtype=tf.float32, shape=(None, num_classes), name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# 预处理：
# 神经网络在训练和测试阶段的预处理方法不同：
# 训练：输入图像随机裁剪，水平翻转，并且用随机值来微调色调，对比度和饱和度，
#      这样就创建了原始图像的随机变体，人为扩充了训练集
# 测试：输入图像根据中心裁剪，并且不作调整

def pre_process_image(image, training):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.

    if training:
        # For training, add the following to the TensorFlow graph.

        # Randomly crop the input image
        image = tf.random_crop(value=image, size=[img_size_cropped, img_size_cropped, num_channels])

        # Randomly flip the image horizontally
        image = tf.image.random_flip_left_right(image)

        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image=image, max_delta=0.05)
        image = tf.image.random_contrast(image=image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image=image, max_delta=0.2)
        image = tf.image.random_saturation(image=image, lower=0.0, upper=2.0)

        # Some of these functions may overflow and result in pixel
        # values beyond the [0, 1] range. It is unclear from the
        # documentation of TensorFlow whether this is intended.
        # A simple solution is to limit the range.

        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)

    else:
        image = tf.image.resize_image_with_crop_or_pad(image=image, target_height=img_size_cropped, target_width=img_size_cropped)

    return image

def pre_process(images, training):
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)

    return images

# 预处理graph
distorted_images = pre_process(images=x, training=True)

def main_network(images, training):
    # Wrap the input images as a Pretty Tensor object
    x_pretty = pt.wrap(images)

    # Pretty Tensor uses special numbers to distinguish between
    # the training and testing phases
    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer

    # Create the convolutional neural network using Pretty Tensor.
    # It is very similar to the previous tutorails, except
    # the use of so-called batch-normalization in the first layer

    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
        y_pred, loss = x_pretty.\
            conv2d(kernel=5, depth=64, name='layer_conv1', batch_normalize=True).\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=5, depth=64, name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=256, name='layer_fc1').\
            fully_connected(size=128, name='layer_fc2').\
            softmax_classifier(num_classes=num_classes, labels=y_true)

    return y_pred, loss

def create_network(training):
    # Wrap the neural network in the scope named 'network'
    # Create new variables during training, and re-use during testing.

    with tf.variable_scope('network', reuse=not training):
        # Just rename the input placeholder variable for convenience.
        images = x

        # Create TensorFlow graph for pre-processing
        images = pre_process(images=images, training=training)

        # Create TensorFlow graph for the main processing
        y_pred, loss = main_network(images, training)

    return y_pred, loss

# 为训练阶段创建神经网络
# 首先创建一个保存当前优化迭代次数的TensorFlow
# trainable=False表示TensorFlow不会优化此变量

global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)

# 创建训练用的神经网络
_, loss = create_network(training=True)

# 创建最小化loss函数的优化器，同时将global_step传给优化器，这样每次迭代它都减一
optimizer = tf.train.AdamOptimizer(learning_rate=0.4).minimize(loss, global_step=global_step)

# 创建测试阶段的神经网络
y_pred, _ = create_network(training=False)
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

# 获取权重
def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'weights' in the scope
    # with the given layer_name

    with tf.variable_scope("network/" + layer_name, reuse=True):
        variable = tf.get_variable('weights')

    return variable

weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')

# 获取layer的输出
def get_layer_output(layer_name):
    # The name of the last operation of the convolutional layer.
    # This assumes you are using relu as the activation-function.
    tensor_name = "network/" + layer_name + "/Relu:0"

    tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)

    return tensor

output_conv1 = get_layer_output(layer_name='layer_conv1')
output_conv2 = get_layer_output(layer_name='layer_conv2')

session = tf.Session()

save_dir = 'checkpoints/cifar'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'cifar10_cnn')

try:
    print("Trying to restore last checkpoint ...")

    # Use TensorFlow to find the latest checkpoint - if any
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)

    # Try and load the data in the checkpoint
    saver.restore(session, save_path=last_chk_path)

    # If we get to this point, the checkpoint was successfully loaded.
    print("Restored checkpoint from: ", last_chk_path)

except:
    # If the above failed for some reason, simply
    # initialize all the variables for the TensorFlow graph
    print("Failed to restore checkpoint. Initializing variables instead.")
    session.run(tf.global_variables_initializer())

train_batch_size = 64

def random_batch():
    num_images = len(images_train)

    idx = np.random.choice(num_images, size=train_batch_size, replace=False)

    x_batch = images_train[idx, :, :, :]
    y_batch = labels_train[idx, :]

    return x_batch, y_batch

def optimize(num_iterations):
    start_time = time.time()

    for i in range(num_iterations):
        x_batch, y_batch = random_batch()

        feed_dict = {x: x_batch, y_true: y_batch}

        i_global, _ = session.run([global_step, optimizer], feed_dict=feed_dict)

        if (i_global % 100 == 0) or (i == num_iterations - 1):
            batch_acc = session.run(accuracy, feed_dict=feed_dict)

            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

        if (i_global % 1000 == 0) or (i == num_iterations - 1):
            saver.save(session, save_path=save_path, global_step=global_step)
            print("Saved checkpoint.")

    end_time = time.time()

    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = images_test[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = cls_test[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

def plot_confusion_matrix(cls_pred):
    cm = confusion_matrix(y_true=cls_test, y_pred=cls_pred)
    for i in range(num_classes):
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))

batch_size = 256
def predict_cls(images, labels, cls_true):
    num_images = len(images)

    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    i = 0

    while i < num_images:
        j = min(i + batch_size, num_images)

        feed_dict = {x: images[i:j, :], y_true: labels[i:j, :]}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    correct = (cls_true == cls_pred)

    return correct, cls_pred

def predict_cls_test():
    return predict_cls(images=images_test, labels=labels_test, cls_true=cls_test)


def classification_accuracy(correct):
    # When averaging a boolean array, False means 0 and True means 1.
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.

    # Return the classification accuracy
    # and the number of correct classifications.
    return correct.mean(), correct.sum()


def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_cls_test()

    # Classification accuracy and the number of correct classifications.
    acc, num_correct = classification_accuracy(correct)

    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

# 绘制卷积层权重
def plot_conv_weights(weights, input_channel=0):
    w = session.run(weights)

    print("Min:  {0:.5f}, Max:   {1:.5f}".format(w.min(), w.max()))
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)
    abs_max = max(abs(w_min), abs(w_max))

    num_filters = w.shape[3]
    num_grids = int(math.ceil(math.sqrt(num_filters)))

    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_filters:
            # Get the weights for the i'th filter of the input channel.
            # The format of this 4-dim tensor is determined by the
            # TensorFlow API. See Tutorial #02 for more details.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=-abs_max, vmax=abs_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# 绘制卷积层输出
def plot_layer_output(layer_output, image):
    feed_dict = {x: [image]}

    values = session.run(layer_output, feed_dict=feed_dict)

    values_min = np.min(values)
    values_max = np.max(values)

    # Number of image channels output by the conv. layer.
    num_images = values.shape[3]

    # Number of grid-cells to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = int(math.ceil(math.sqrt(num_images)))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid image-channels.
        if i < num_images:
            # Get the images for the i'th output channel.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, vmin=values_min, vmax=values_max,
                      interpolation='nearest', cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# 输入图像变体
def plot_distorted_image(image, cls_true):
    image_duplicates = np.repeat(image[np.newaxis, :, :, :], 9, axis=0)

    feed_dict = {x: image_duplicates}

    result = session.run(distorted_images, feed_dict=feed_dict)

    plot_images(images=result, cls_true=np.repeat(cls_true, 9))

def get_test_image(i):
    return images_test[i, :, :, :], cls_test[i]

img, cls = get_test_image(16)
plot_distorted_image(img, cls)

optimize(1000)
print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)

plot_conv_weights(weights=weights_conv1, input_channel=0)

plot_conv_weights(weights=weights_conv2, input_channel=1)


def plot_image(image):
    # Create figure with sub-plots.
    fig, axes = plt.subplots(1, 2)

    # References to the sub-plots.
    ax0 = axes.flat[0]
    ax1 = axes.flat[1]

    # Show raw and smoothened images in sub-plots.
    ax0.imshow(image, interpolation='nearest')
    ax1.imshow(image, interpolation='spline16')

    # Set labels.
    ax0.set_xlabel('Raw')
    ax1.set_xlabel('Smooth')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

img, cls = get_test_image(16)
plot_image(img)

# 将原始图像作为神经网络的输入，画出第一个卷积层输出
plot_layer_output(output_conv1, image=img)

# 将原始图像作为神经网络的输入，画出第二个卷积层输出
plot_layer_output(output_conv2, image=img)

label_pred, cls_pred = session.run([y_pred, y_pred_cls],
                                   feed_dict={x: [img]})

# Set the rounding options for numpy.
np.set_printoptions(precision=3, suppress=True)

# Print the predicted label.
print(label_pred[0])

session.close()