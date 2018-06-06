##########################
####### 卷积神经网络集成
####### 使用多个网络，输出平均
##########################

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
data = input_data.read_data_sets('data/MNIST/', one_hot = True)

data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)

# 将训练集和验证集合并
combined_images = np.concatenate([data.train.images, data.validation.images], axis=0)
combined_labels = np.concatenate([data.train.labels, data.validation.labels], axis=0)

# print(combined_images.shape)           # (60000, 784)
# print(combined_labels.shape)           # (60000, 10)

combined_size = combined_images.shape[0]
train_size = int(0.8 * combined_size)
validation_size = combined_size - train_size

# 重新切分
def random_training_set():
    inx = np.random.permutation(combined_size)

    inx_train = inx[0:train_size]
    inx_validation = inx[train_size:]

    x_train = combined_images[inx_train, :]
    y_train = combined_labels[inx_train, :]

    x_validation = combined_images[inx_validation, :]
    y_validation = combined_labels[inx_validation, :]

    return x_train, y_train, x_validation, y_validation

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

def plot_images(images,
                cls_true,
                ensemble_cls_pred=None,
                best_cls_pred=None):
    assert len(images) == len(cls_true)

    fig, axes = plt.subplots(3, 3)

    if ensemble_cls_pred is None:
        hspace=0.3
    else:
        hspace=1.0
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i].reshape(img_shape), cmap='binary')

            if ensemble_cls_pred is None:
                xlabel = "True: {0}".format(cls_true[i])
            else:
                msg = "True: {0}\nEnsemble: {1}\nBest Net: {2}"
                xlabel = msg.format(cls_true[i],
                                    ensemble_cls_pred[i],
                                    best_cls_pred[i])

            ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

x = tf.placeholder(dtype=tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, shape=([-1, img_size, img_size, num_channels]))
y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# implemented by pretty tensor
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

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(max_to_keep=100)
save_dir = 'checkpoints/ensemble/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def get_save_path(net_number):
    return save_dir + 'network' + str(net_number)

session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = 64

def random_batch(x_train, y_train):
    inx = np.random.choice(len(x_train), size=train_batch_size, replace=False)

    x_batch = x_train[inx, :]
    y_batch = y_train[inx, :]

    return x_batch, y_batch

def optimize(num_iterations, x_train, y_train):
    start_time = time.time()

    for i in range(num_iterations):
        x_batch, y_batch = random_batch(x_train, y_train)

        feed_dict = {x: x_batch, y_true: y_batch}

        session.run(optimizer, feed_dict=feed_dict)

        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict)
            msg = "Optimization Iteration: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))

    end_time = time.time()
    time_dif = end_time - start_time

    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

num_networks = 5
num_iterations = 500

'''
if True:
    for i in range(num_networks):
        print("Neural Network: {0}".format(i))

        x_train, y_train, _, _ = random_training_set()

        session.run(tf.global_variables_initializer())
        optimize(num_iterations, x_train, y_train)

        saver.save(sess=session, save_path=get_save_path(i))
        print()
'''

batch_size = 128

def predict_labels(images):
    num_images = len(images)

    pred_labels = np.zeros(shape=(num_images, num_classes), dtype=np.float32)

    i = 0
    while i < num_images:
        j = min(i + batch_size, num_images)

        batch_images = images[i:j, :]

        feed_dict = {x: batch_images}

        pred_labels[i:j, :] = session.run(y_pred, feed_dict=feed_dict)

        i = j

    return pred_labels

def correct_prediction(images, cls_true):
    pred_labels = predict_labels(images)
    pred_cls = np.argmax(pred_labels, axis=1)

    correct = (pred_cls == cls_true)

    return correct

def test_correct():
    return correct_prediction(images=data.test.images, cls_true=data.test.cls)

def validation_correct():
    return correct_prediction(images=data.validation.images, cls_true=data.validation.cls)

def classification_accuracy(correct):
    return correct.mean()

def test_accuracy():
    correct = test_correct()
    return classification_accuracy(correct)

def validation_accuracy():
    correct = validation_correct()
    return classification_accuracy(correct)

def ensemble_predictions():
    pred_labels = []
    test_accs = []
    validation_accs = []

    for i in range(num_networks):
        saver.restore(sess=session, save_path=get_save_path(i))

        test_acc = test_accuracy()
        test_accs.append(test_acc)

        validation_acc = validation_accuracy()
        validation_accs.append(validation_acc)

        msg = "Network: {0}, Accuracy on Validation-Set: {1:.4f}, Test-Set: {2:.4f}"
        print(msg.format(i, validation_acc, test_acc))

        pred_label = predict_labels(images=data.test.images)

        pred_labels.append(pred_label)

    return np.array(pred_labels), \
           np.array(test_accs), \
           np.array(validation_accs)

pred_labels, test_accuracies, val_accuracies = ensemble_predictions()
print("Mean test-set accuracy: {0:.4f}".format(np.mean(test_accuracies)))
print("Min test-set accuracy:  {0:.4f}".format(np.min(test_accuracies)))
print("Max test-set accuracy:  {0:.4f}".format(np.max(test_accuracies)))

print(pred_labels.shape)                    # (5, 10000, 10)

ensemble_pred_labels = np.mean(pred_labels, axis=0)
print(ensemble_pred_labels.shape)           # (10000, 10)

ensemble_pred_cls = np.argmax(ensemble_pred_labels, axis=1)
print(ensemble_pred_cls)                    # (10000, )

ensemble_correct = (ensemble_pred_cls == data.test.cls)
ensemble_incorrect = np.logical_not(ensemble_correct)

# 列出每个模型的准确率
print(test_accuracies)

# 最佳模型
best_net = np.argmax(test_accuracies)
best_net_pred_labels = pred_labels[best_net, :, :]
best_net_cls_pred = np.argmax(best_net_pred_labels, axis=1)
best_net_correct = (best_net_cls_pred == data.test.cls)
best_net_incorrect = np.logical_not(best_net_correct)
ensemble_better = np.logical_and(best_net_incorrect,
                                 ensemble_correct)
best_net_better = np.logical_and(best_net_correct,
                                 ensemble_incorrect)

def plot_images_comparison(idx):
    plot_images(images=data.test.images[idx, :],
                cls_true=data.test.cls[idx],
                ensemble_cls_pred=ensemble_pred_cls[idx],
                best_cls_pred=best_net_cls_pred[idx])


def print_labels(labels, idx, num=1):
    # Select the relevant labels based on idx.
    labels = labels[idx, :]

    # Select the first num labels.
    labels = labels[0:num, :]

    # Round numbers to 2 decimal points so they are easier to read.
    labels_rounded = np.round(labels, 2)

    # Print the rounded labels.
    print(labels_rounded)

def print_labels_ensemble(idx, **kwargs):
    print_labels(labels=ensemble_pred_labels, idx=idx, **kwargs)

def print_labels_best_net(idx, **kwargs):
    print_labels(labels=best_net_pred_labels, idx=idx, **kwargs)

def print_labels_all_nets(idx):
    for i in range(num_networks):
        print_labels(labels=pred_labels[i, :, :], idx=idx, num=1)

plot_images_comparison(idx=ensemble_better)
print_labels_ensemble(idx=ensemble_better, num=1)
print_labels_best_net(idx=ensemble_better, num=1)
print_labels_all_nets(idx=ensemble_better)
plot_images_comparison(idx=best_net_better)
print_labels_ensemble(idx=best_net_better, num=1)
print_labels_best_net(idx=best_net_better, num=1)
print_labels_all_nets(idx=best_net_better)
