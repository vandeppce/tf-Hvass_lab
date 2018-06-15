import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras import backend as K
from keras.objectives import categorical_crossentropy

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

x = tf.placeholder(dtype=tf.float32, shape=(None, img_size_flat), name='x')
x_image = tf.reshape(tensor=x, shape=(-1, img_size, img_size, num_channels), name='x_image')
y_true = tf.placeholder(dtype=tf.float32, shape=(None, num_classes), name='y_true')
y_true_cls = tf.argmax(input=y_true, axis=1)

noise_limit = 0.35
noise_l2_weight = 0.02

ADVERSARY_VARIABLES = 'adversari_variables'
collections = [tf.GraphKeys.GLOBAL_VARIABLES, ADVERSARY_VARIABLES]

x_noise = tf.Variable(tf.zeros(shape=(img_size, img_size, num_channels)), collections=collections, trainable=False, name='x_noise')
x_noise_clip = tf.clip_by_value(t=x_noise, clip_value_min=-noise_limit, clip_value_max=noise_limit, name='x_noise_clip')

x_noisy_image = x_noise_clip + x_image
x_noisy_image = tf.clip_by_value(x_noisy_image, 0.0, 1.0)

x_deal = Conv2D(filters=16, kernel_size=5, strides=1, padding='same', name='layer_conv1')(x_noisy_image)
x_deal = MaxPooling2D(pool_size=2, strides=2, name='max_pool1')(x_deal)
x_deal = Conv2D(filters=36, kernel_size=5, strides=1, padding='same', name='layer_conv2')(x_deal)
x_deal = MaxPooling2D(pool_size=2, strides=2, name='max_pool2')(x_deal)
x_deal = Flatten()(x_deal)
x_deal = Dense(units=128, activation='relu')(x_deal)
y_pred = Dense(units=num_classes, activation='softmax')(x_deal)

loss = tf.reduce_mean(categorical_crossentropy(y_true=y_true, y_pred=y_pred))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

# print(tf.global_variables())
# print(tf.trainable_variables())
# print(tf.get_collection(key=ADVERSARY_VARIABLES))

l2_loss_noise = noise_l2_weight * tf.nn.l2_loss(x_noise)
loss_adversary = l2_loss_noise + loss

adversary_variables = tf.get_collection(key=ADVERSARY_VARIABLES)
optimizer_adversary = tf.train.AdamOptimizer(1e-2).minimize(loss_adversary, var_list=adversary_variables)

y_pred_cls = tf.argmax(y_pred, axis=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
# writer = tf.summary.FileWriter('summary', graph=session.graph)
# writer.close()
session.run(tf.global_variables_initializer())

def init_noise():
    session.run(tf.variables_initializer([x_noise]))

train_batch_size = 64

def optimize(num_iterations, adversary_target_cls=None):
    for i in range(num_iterations):
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        if adversary_target_cls is not None:
            y_true_batch = np.zeros_like(y_true_batch)
            y_true_batch[:, adversary_target_cls] = 1.0

        feed_dict_train = {x: x_batch, y_true: y_true_batch}

        if adversary_target_cls is None:
            session.run(optimizer, feed_dict=feed_dict_train)
        else:
            session.run(optimizer_adversary, feed_dict=feed_dict_train)

        if (i % 100 == 0) or (i == num_iterations - 1):
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i, acc))

def get_noise():
    noise = session.run(x_noise)
    return np.squeeze(noise)

test_batch_size = 256

def print_test_accuracy():
    num_test = len(data.test.images)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    i = 0

    while i < num_test:
        j = min(i + test_batch_size, num_test)

        images = data.test.images[i:j, :]
        labels = data.test.labels[i:j, :]

        feed_dict = {x: images,
                     y_true: labels}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    cls_true = data.test.cls
    correct = (cls_true == cls_pred)
    correct_sum = correct.sum()

    acc = float(correct_sum) / num_test

    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

# 正常训练
optimize(num_iterations=1000)
print_test_accuracy()

# 训练对抗噪声
optimize(num_iterations=1000, adversary_target_cls=3)
print_test_accuracy()

# 免疫
init_noise()
def make_immune(target_cls, num_iterations_adversay=500, num_iterations_immune=300):
    print("Target-class:", target_cls)
    print("Finding adversarial noise ...")

    optimize(num_iterations=num_iterations_adversay, adversary_target_cls=target_cls)
    print()

    print_test_accuracy()
    print()

    print("Making the neural network immune to the noise ...")

    optimize(num_iterations=num_iterations_immune)
    print()

    print_test_accuracy()

make_immune(target_cls=3)
session.close()