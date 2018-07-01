import tensorflow as tf
import numpy as np
import scipy.io
import os

VGG_FILENAME = 'imagenet-vgg-verydeep-19.mat'
VGG_FILEPATH = 'vgg/'

class CONFIG(object):
    IMAGE_WIDTH = 400
    IMAGE_HEIGHT = 300
    COLOR_CHANNELS = 3
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

class VGG(CONFIG):
    def __init__(self):
        self.layers = scipy.io.loadmat(os.path.join(VGG_FILEPATH, VGG_FILENAME))['layers']
        self.mean_pixels = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

    def _weights(self, layer_idx, expected_layer_name):
        """
        获取layer层的指定权重

        :param layer_idx: VGG中的layer_id
        :param expected_layer_name: layer命名

        :return: pre-trained的权重W和b
        """

        W = self.layers[0][layer_idx][0][0][0][0][0]
        b = self.layers[0][layer_idx][0][0][0][0][1]

        layer_name = self.layers[0][layer_idx][0][0][-2]

        return W, b

    def conv2d_relu(self, prev_layer, layer_idx, layer_name):
        """
        采用relu作为激活函数的卷积层

        :param prev_layer: 前一层输出
        :param layer_idx: layer_id
        :param expeccted_layer_name: layer命名

        :return:
        """

        with tf.variable_scope(layer_name):
            # 获取权重
            W, b = self._weights(layer_idx=layer_idx, expected_layer_name=layer_name)

            # 转化为constant, 不需要训练
            W = tf.constant(W, name='weights')
            b = tf.constant(b, name='bias')

            # 卷积层
            conv2d = tf.nn.conv2d(input=prev_layer,
                                  filter=W,
                                  strides=[1, 1, 1, 1],
                                  padding='SAME')

            out = tf.nn.relu(conv2d + b)

        return out

    def avgpool(self, prev_layer, layer_name):
        with tf.variable_scope(layer_name):
            out = tf.nn.avg_pool(value=prev_layer,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME')

        return out

    def load(self):
        graph = {}
        graph['input'] = tf.Variable(np.zeros((1, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.COLOR_CHANNELS)),
                                     dtype='float32')
        graph['conv1_1'] = self.conv2d_relu(graph['input'], 0, 'conv1_1')
        graph['conv1_2'] = self.conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
        graph['avgpool1'] = self.avgpool(graph['conv1_2'], 'avgpoo1')
        graph['conv2_1'] = self.conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
        graph['conv2_2'] = self.conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
        graph['avgpool2'] = self.avgpool(graph['conv2_2'], 'avgpool2')
        graph['conv3_1'] = self.conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
        graph['conv3_2'] = self.conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
        graph['conv3_3'] = self.conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
        graph['conv3_4'] = self.conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
        graph['avgpool3'] = self.avgpool(graph['conv3_4'], 'avgpool3')
        graph['conv4_1'] = self.conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
        graph['conv4_2'] = self.conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
        graph['conv4_3'] = self.conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
        graph['conv4_4'] = self.conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
        graph['avgpool4'] = self.avgpool(graph['conv4_4'], 'avgpool4')
        graph['conv5_1'] = self.conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
        graph['conv5_2'] = self.conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
        graph['conv5_3'] = self.conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
        graph['conv5_4'] = self.conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
        graph['avgpool5'] = self.avgpool(graph['conv5_4'], 'avgpool5')

        return graph