"""
Early-stopping: 当验证集上分类准确率提高时，保存神经网络的变量。
                如果经过1000次迭代还不能提升性能，就终止优化。
                然后重新载入在验证集上表现最好的变量。
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os

import prettytensor as pt
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

def plot_images(images, cls_true, cls_pred):
    assert len(images) == len(cls_true) == 9

    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

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

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

x_pretty = pt.wrap(x_image)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        conv2d(kernel=5, depth=16, name='layer_conv1').\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=36, name='layer_conv2').\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)

def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('weights')

    return variable

weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_true_cls, y_pred_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Saver
saver = tf.train.Saver()

save_dir = 'checkpoints/'

# 如果文件不存在则创建
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'best_validation')

# 运行tf
session = tf.Session()

# 初始化变量
session.run(tf.global_variables_initializer())

# 优化迭代
train_batch_size = 64

# 每迭代100次，计算一次验证集准确率。
# 如果超过了1000次迭代验证集准确率还没有提升，就停止优化

# Best validation accuracy seen so far
best_validation_accuracy = 0.0

# Iteration-number for last improvement to validation accuracy.
last_improvement = 0

# Stop optimization if no improvement found in this many iterations
require_improvement = 1000

total_iterations = 0

def optimize(num_iterations):
    global total_iterations
    global best_validation_accuracy
    global last_improvement

    start_time = time.time()

    for i in range(num_iterations):
        # Increase the total number of iterations performed.
        # It is easier to update it in each iteration because
        # we need this number several times in the following

        total_iterations += 1
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        feed_dict = {x: x_batch, y_true: y_true_batch}

        session.run(optimizer, feed_dict=feed_dict)

        # print status every 100 iterations and after last iteration
        if (total_iterations % 100 == 0) or (i == (num_iterations - 1)):
            # Calculate the accuracy on the training-batch
            acc_train = session.run(accuracy, feed_dict=feed_dict)

            # Calculate the accuracy on the validation-set
            # The function returns 2 values but we only need the first
            acc_validation, _ = validation_accuracy()

            # If validation accuracy is an improvement over best-known
            if acc_validation > best_validation_accuracy:
                # Update the best known validation accuracy
                best_validation_accuracy = acc_validation

                # Set the iteration for the last improvement to current
                last_improvement = total_iterations

                # Save all variables of the TensofFlow graph to file
                saver.save(sess=session, save_path=save_path)

                # A string to be printed below, shows improvement foung
                improved_str = "*"
            else:
                # An empty string to be printed below
                # Shows that no improvement was found
                improved_str = ''

            # Status-message for printing
            msg = "Iter: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, validation acc: {2:>6.1%} {3}"

            print(msg.format(i + 1, acc_train, acc_validation, improved_str))

        # If no improvement found in the required number of iterations
        if total_iterations - last_improvement > require_improvement:
            print("No improvement found in a while, stopping optimization.")

            break

    end_time = time.time()

    time_dif = end_time - start_time

    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.
    # cls_pred is an array of the predicted class-number for all images in the test-set

    # correct is a boolen array whether the predicted class is equal to the true class for each image in the test-set

    # Negate the boolen array

    incorrect = (correct == False)

    images = data.test.images[incorrect]

    cls_pred = cls_pred[incorrect]

    cls_true = data.test.cls[incorrect]

    plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])

def plot_confusion_matrix(cls_pred):
    cls_true = data.test.cls

    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)

    print(cm)

    plt.matshow(cm)

    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()

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

    correct = (cls_pred == cls_true)

    return correct, cls_pred

def predict_cls_test():
    return predict_cls(images=data.test.images,
                       labels=data.test.labels,
                       cls_true=data.test.cls)

def predict_cls_validation():
    return predict_cls(images=data.validation.images,
                       labels=data.validation.labels,
                       cls_true=data.validation.cls)

# 分类准确率
def cls_accuracy(correct):
    correct_sum = correct.sum()

    acc = float(correct_sum) / len(correct)

    return acc, correct_sum

# 验证集准确率
def validation_accuracy():
    correct, _ = predict_cls_validation()

    return cls_accuracy(correct)

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    correct, cls_pred = predict_cls_test()

    acc, num_correct = cls_accuracy(correct)

    num_images = len(correct)

    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    if show_example_errors:
        print("Examples errors: ")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    if show_confusion_matrix:
        print("Confusion Matrix: ")
        plot_confusion_matrix(cls_pred=cls_pred)

def plot_conv_weights(weights, input_channel=0):

    w = session.run(weights)

    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))

    w_min = np.min(w)
    w_max = np.max(w)

    num_filters = w.shape[3]

    num_grids = np.int(math.ceil(math.sqrt(num_filters)))

    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = w[:, :, input_channel, i]

            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


print_test_accuracy()

plot_conv_weights(weights=weights_conv1)

optimize(num_iterations=3000)

print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)

plot_conv_weights(weights=weights_conv1)



# 恢复变量
saver.restore(sess=session, save_path=save_path)

print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)

plot_conv_weights(weights=weights_conv1)

session.close()