#################################################################
# 教程#11通过单张图片的优化寻找对抗噪声，所以噪声可能不是通用的。
# 本教程找到那些针对所有输入图像的噪声，现在对抗噪声对人眼是清晰可见的，
# 但是人类还是能辨认出数字，然而神经网络几乎将所有图像误分类。
# #11使用numpy做优化，#12将会直接在tf里实现优化过程，这样会更快速，
# 尤其是在使用GPU时，因为不用每次迭代都在GPU中拷贝数据。

# 在这个NN中有两个单独的优化程序。
# 首先，优化NN来分类图像（常规优化），一旦精确率足够高，切换到第二个优化。
# 第二个优化程序用来寻找单一模式的对抗噪声，使得所有输入图像都被误分类。
# 这两个优化程序完全独立，第一个只修改神经网络的变量，第二个只修改对抗噪声
#################################################################

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

import prettytensor as pt

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

data.test.cls = np.argmax(data.test.labels, axis=1)

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

def plot_images(images, cls_true, cls_pred=None, noise=0.0):
    assert len(images) == len(cls_true) == 9

    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        image = images[i].reshape(img_shape)

        image += noise

        image = np.clip(image, 0.0, 1.0)

        ax.imshow(image, cmap='binary', interpolation='nearest')

        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, shape=(-1, img_size, img_size, num_channels))
y_true = tf.placeholder(tf.float32, [None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# 对抗干扰
# 输入图像的像素值在0.0到1.0之间，对抗噪声是在输入图像上添加或删除的数值，界限设置为0.35

noise_limit = 0.35

# 对抗噪声的优化器会试图最小化两个损失：
# 1. 神经网络常规的损失，会使我们找到目标类型分类准确率最高的噪声
# 2. L2-loss噪声，保持尽可能低的噪声
# 下面的权重决定了与常规的损失度量相比，L2-loss的重要性，通常接近0的L2权重表现更好

noise_l2_weight = 0.02

# 为噪声创建变量时，必须告诉TensorFlow它属于哪一个变量集合，
# 这样，后面就能通知两个优化器更新哪些变量

ADVERSARY_VARIABLES = 'adversary_variables'

# 接着，创建噪声变量所属集合的列表，如果将噪声变量添加到集合tf.GraphKeys.GLOBAL_VARIABLES中，
# 它就会和TensorFlow graph中的其他变量一起被初始化，但不会被优化

collections = [tf.GraphKeys.GLOBAL_VARIABLES, ADVERSARY_VARIABLES]

# 为对抗噪声添加新的变量，初始化为0，不可训练，这样就不会和NN中的其他变量一起训练
x_noise = tf.Variable(tf.zeros(shape=(img_size, img_size, num_channels)), name='x_noise', trainable=False, collections=collections)
x_noise_clip = tf.clip_by_value(x_noise, -noise_limit, noise_limit)

x_noisy_image = x_noise_clip + x_image
x_noisy_image = tf.clip_by_value(x_noisy_image, 0.0, 1.0)

x_pretty = pt.wrap(x_noisy_image)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        conv2d(kernel=5, depth=16, name='layer_conv1').\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=36, name='layer_conv2').\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)

# print(tf.global_variables())
# print(tf.trainable_variables())

# 神经网络变量优化
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# 对抗噪声优化
adversary_variables = tf.get_collection(ADVERSARY_VARIABLES)

# print(adversary_variables)

# 损失函数
l2_loss_noise = noise_l2_weight * tf.nn.l2_loss(x_noise)
loss_adversary = loss + l2_loss_noise

# 现在为对抗噪声创建优化器，由于优化器并不会更新神经网络的所有变量，所以必须给出一个更新变量的列表
optimizer_adversary = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss_adversary, var_list=adversary_variables)

# 性能度量
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())
# writer = tf.summary.FileWriter('summary', session.graph)
# writer.close()
train_batch_size = 64

# 优化函数
# 与之前的神经网络相比，多了一个对抗目标类别参数，当目标类别设置为整数时，
# 将会用它代替训练集中的真实类别号，采用对抗优化器代替常规优化器。

def optimize(num_iterations, adversary_target_cls=None):
    start_time = time.time()

    for i in range(num_iterations):
        x_batch, y_true_batch = data.train.next_batch(batch_size=train_batch_size)

        # If we are searching for the adversarial noise, then
        # use the adversarial target-class instead.
        if adversary_target_cls is not None:
            # Set all the class-labels to zero
            y_true_batch = np.zeros_like(y_true_batch)     # 按照y_true_batch的形状生成全0数组

            # Set the element for the adversarial target-class to 1
            y_true_batch[:, adversary_target_cls] = 1.0

        feed_dict_train = {x: x_batch, y_true: y_true_batch}

        # If doing normal optimization of the neural network
        if adversary_target_cls is None:
            session.run(optimizer, feed_dict=feed_dict_train)
        else:
            session.run(optimizer_adversary, feed_dict=feed_dict_train)

            session.run(x_noise_clip)

        if (i % 100 == 0) or (i == num_iterations - 1):
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            msg = "Optimization Iterations: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i, acc))

    end_time = time.time()

    time_dif = end_time - start_time
    print("Time Usage: " + str(timedelta(seconds=int(round(time_dif)))))

def init_noise():
    session.run(tf.variables_initializer([x_noise]))

def get_noise():
    noise = session.run(x_noise)
    return np.squeeze(noise)

def plot_noise():
    noise = get_noise()

    print("Noise:")
    print("- Min:", noise.min())
    print("- Max:", noise.max())
    print("- Std:", noise.std())

    plt.imshow(noise, interpolation='nearest', cmap='seismic', vmin=-1.0, vmax=1.0)


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
    images = data.test.images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]

    # Get the adversarial noise from inside the TensorFlow graph.
    noise = get_noise()

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9],
                noise=noise)


def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.test.cls

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

# 常规优化
# 此时对抗噪声无效，因为初始化为0，且未优化
# optimize(num_iterations=1000)
# print_test_accuracy(show_example_errors=True)

# 对抗优化
# optimize(num_iterations=1000, adversary_target_cls=3)
# plot_noise()
# print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)

# 所有目标类别的对抗噪声
def fine_all_noise(num_iterations=1000):
    all_noise = []
    for i in range(num_classes):
        print("Finding adversarial noise for target-class:", i)

        init_noise()

        optimize(num_iterations=num_iterations, adversary_target_cls=i)

        noise = get_noise()
        all_noise.append(noise)

        print()
    return all_noise


def plot_all_noise(all_noise):
    # Create figure with 10 sub-plots.
    fig, axes = plt.subplots(2, 5)
    fig.subplots_adjust(hspace=0.2, wspace=0.1)

    # For each sub-plot.
    for i, ax in enumerate(axes.flat):
        # Get the adversarial noise for the i'th target-class.
        noise = all_noise[i]

        # Plot the noise.
        ax.imshow(noise,
                  cmap='seismic', interpolation='nearest',
                  vmin=-1.0, vmax=1.0)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(i)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# all_noise = fine_all_noise(num_iterations=300)
# plot_all_noise(all_noise)

# 对抗噪声免疫
# 首先寻找噪声，再利用噪声训练网络

def make_immune(target_cls, num_iterations_adversary=500, num_iterations_immune=200):
    print("Target-class: ", target_cls)
    print("Finding adversarial noise ...")

    optimize(num_iterations=num_iterations_adversary, adversary_target_cls=target_cls)
    print()

    print_test_accuracy(show_example_errors=False, show_confusion_matrix=False)
    print()

    print("Making the neural network immune to the noise ...")
    # Note that the adversarial noise has not been reset to zero
    # so the x_noise variable still holds the noise
    # So we are training the neural network to ignore the noise
    optimize(num_iterations=num_iterations_immune)
    print()

    print_test_accuracy(show_confusion_matrix=False, show_example_errors=False)

# 对目标类型3的噪声免疫
make_immune(target_cls=3)

# 再运行一遍，由于所有的变量均为重新初始化，所以相当于在上一次immune的基础上运行
make_immune(target_cls=3)
make_immune(target_cls=3)

# 噪声图像的性能
print_test_accuracy(show_example_errors=True)

# 干净图像的性能
init_noise()
print_test_accuracy(show_example_errors=True)
session.close()