####################################################################
# Using Inception V3 for trainsfer learning

# Inception 无法对人物人类，因为训练集有很多易混淆的特征。
# 我们可以使用Inception来提取特征，因此可以用其他训练集训练，但这时间代价昂贵。
# 相反，可以复用预训练的Inception模型，然后替换掉最后做分类的那一层。
# 这种方法叫做迁移学习。

# 在Inception中输入并处理图像，在模型最终的分类层之前，
# 将Transfer-values保存到缓存文件中
# 随后基于Transfer-values来学习分类图像
####################################################################

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os

import prettytensor as pt

import cifar10
from cifar10 import num_classes

import inception
from inception import transfer_values_cache

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

# cifar10.maybe_download_and_extract()

class_names = cifar10.load_class_names()
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

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

#inception.maybe_download()
model = inception.Inception()

# 设置训练集和测试集缓存文件的目录
file_path_cache_train = os.path.join(cifar10.data_path, 'inception_cifar10_train.pkl')
file_path_cache_test = os.path.join(cifar10.data_path, 'inception_cifar10_test.pkl')

print("Processing Inception transfer-values for training images ...")

# Scale images because Inception need pixels to be between 0 and 255
# while the CIFAR-10 functions return pixels between 0.0 and 1.0

images_scaled = images_train * 255.0
transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train, model=model, images=images_scaled)

print("Processing Inception transfer-values for testing images ...")

images_scaled = images_test * 255.0
transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test, model=model, images=images_scaled)

print(transfer_values_train.shape)
print(transfer_values_test.shape)

# 绘制transfer-value
def plot_transfer_values(i):
    print("Input image:")

    plt.imshow(images_test[i], interpolation='nearest')
    plt.imshow()

    print("Transfer-values for the image using Inception model:")

    img = transfer_values_test[i]
    img = img.reshape((32, 64))

    plt.imshow(img, interpolation='nearest', cmap='Reds')
    plt.show()

plot_transfer_values(i=16)
plot_transfer_values(i=17)

# 利用PCA将transfer-values的数据维度从2048维降到2维
pca = PCA(n_components=2)

# 选择3000个样本
transfer_values=transfer_values_train[0:3000]             # shape: (3000, 2048)
cls = cls_train[0:3000]

# 利用PCA降维
transfer_values_reduced = pca.fit_transform(transfer_values)   # shape: (3000, 2)

# 绘制降维后的transfer-value
def plot_scatter(values, cls):
    cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))

    colors = cmap[cls]

    x = values[:, 0]
    y = values[:, 1]

    plt.scatter(x, y, color=colors)
    plt.show()

plot_scatter(transfer_values_reduced, cls)

# transfer-values的t-SNE分析结果
# t-SNE也是一种降维方法，不过速度很慢，因此先用PCA将维度降到50

pca = PCA(n_components=50)
transfer_values_50d = pca.fit_transform(transfer_values)

tsne = TSNE(n_components=2)
transfer_values_reduced = tsne.fit_transform(transfer_values_50d)

# 画出t-SNE的结果，相比PCA，具有更好的分离度
plot_scatter(transfer_values_reduced, cls)

# 创建新分类器，将Inception模型中的transfer-values作为输入，输出CIFAR-10的预测类别
# 首先找到transfer-values的数组长度，是保存在Inception模型中的一个变量

transfer_len = model.transfer_len

x = tf.placeholder(dtype=tf.float32, shape=[None, transfer_len], name='x')
y_true = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# pretty tensor
x_pretty = pt.wrap(x)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        fully_connected(size=1024, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)

global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step)

y_pred_cls = tf.argmax(y_pred, dimension=1)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = 64

def random_batch():
    num_images = len(transfer_values_train)

    idx = np.random.choice(num_images, size=train_batch_size, replace=False)
    x_batch = transfer_values_train[idx]
    y_batch = labels_train[idx]

    return x_batch, y_batch

def optimize(num_iterations):
    start_time = time.time()

    for i in range(num_iterations):
        x_batch, y_batch = random_batch()

        feed_dict_train = {x: x_batch, y_true: y_batch}

        i_global, _ = session.run([global_step, optimizer], feed_dict=feed_dict_train)

        if (i_global % 100 == 0) or (i == num_iterations - 1):
            batch_acc = session.run(accuracy, feed_dict=feed_dict_train)

            msg = "Global step: {0:>6}, Training Batch Acc: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)

    images = images_test[incorrect]
    cls_pred = images_test[incorrect]
    cls_true = images_test[incorrect]

    n = min(9, len(images))

    plot_images(images[0:n], cls_true=cls_true[0:n], cls_pred=cls_pred[0:n])

def plot_confusion_matrix(cls_pred):
    cm = confusion_matrix(y_true=cls_test, y_pred=cls_pred)

    for i in range(num_classes):
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    class_numbers = ["({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))

batch_size = 256
def predict_cls(transfer_values, labels, cls_true):
    num_images = len(transfer_values)

    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    i = 0
    while i < num_images:
        j = min(i + batch_size, num_images)
        feed_dict = {x: transfer_values[i:j, :], y_true: labels[i:j, :]}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    correct = (cls_true == cls_pred)

    return correct, cls_pred

def predict_cls_test():
    return predict_cls(transfer_values=transfer_values_test, labels=labels_test, cls_true=cls_test)

def classification_accuracy(correct):
    return correct.mean(), correct.sum()

def print_test_accuracy(show_example_errors=False, show_confusion_matrix=False):
    correct, cls_pred = predict_cls_test()

    acc, num_correct = classification_accuracy(correct)

    num_images = len(correct)
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    if show_confusion_matrix:
        plot_confusion_matrix(cls_pred=cls_pred)

# 测试
# 优化前
print_test_accuracy(show_example_errors=False, show_confusion_matrix=False)

# 优化10000次
optimize(10000)
print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)

# 关闭session
# 注意需要关闭两个session，每个模型各一个
model.close()
session.close()