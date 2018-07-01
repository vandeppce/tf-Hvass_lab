import os
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import scipy.misc

import vgg19

class StyleTransfer(vgg19.CONFIG):
    def __init__(self, content_img, style_img, img_width, img_height):
        # 获取基本信息
        self.content_name = str(content_img.split("/")[-1].split(".")[0])
        self.style_name = str(style_img.split("/")[-1].split(".")[0])
        self.img_width = img_width
        self.img_height = img_height

        self.content_img = self.get_resized_img(content_img, img_width, img_height)
        self.style_img = self.get_resized_img(style_img, img_width, img_height)
        self.init_img = self.generate_noise_img(self.content_img)

        # 定义提取特征的层
        self.content_layer = "conv4_2"
        self.style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

        # 定义权重
        self.content_w = 0.001
        self.style_w = 1

        # 不同style layers的权重，层数越深权重越大
        self.style_layer_w = [0.5, 1.0, 1.5, 3.0, 4.0]

        # global step和学习率
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")  # global step
        self.lr = 2.0

    def save_image(self, path, image):
        image = image + self.MEANS
        image = np.clip(image[0], 0, 255).astype('uint8')
        scipy.misc.imsave(path, image)

    def get_resized_img(self, img_path, width, height):
        image = Image.open(img_path)

        # PIL is column major so you have to swap the places of width and height
        image = ImageOps.fit(image, (width, height), Image.ANTIALIAS)
        image_dirs = img_path.split('/')
        image_dirs[-1] = 'resized_' + image_dirs[-1]
        out_path = '/'.join(image_dirs)
        if not os.path.exists(out_path):
            image.save(out_path)
        image = np.asarray(image, np.float32)

        return np.expand_dims(image, 0)

    def generate_noise_img(self, content_img, noise_ratio=0.6):
        noise_image = np.random.uniform(-20, 20, content_img.shape).astype(np.float32)
        return noise_image * noise_ratio + content_img * (1 - noise_ratio)

    # 加载模型
    def load_vgg(self):
        self.model = vgg19.VGG()
        self.graph = self.model.load()

        self.content_img -= self.model.mean_pixels
        self.style_img -= self.model.mean_pixels

    def _content_loss(self):
        session.run(self.graph['input'].assign(self.content_img))
        content_tensor = self.graph[self.content_layer]

        a_C = session.run(content_tensor)
        a_G = content_tensor

        m, n_H, n_W, n_C = a_G.get_shape().as_list()

        # Reshape a_C and a_G
        a_C_unrolled = tf.transpose(tf.reshape(a_C, [n_H * n_W, n_C]))
        a_G_unrolled = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

        # compute the cost with tensorflow
        J_content = 1. / (4. * n_H * n_W * n_C) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))

        return J_content

    def _gram_matrix(self, A, gshape):
        m, n_H, n_W, n_C = gshape

        A = tf.reshape(tensor=A, shape=(n_H * n_W, n_C))
        return tf.matmul(tf.transpose(A), A)

    def _single_style_loss(self, layer_idx):
        session.run(self.graph['input'].assign(self.style_img))
        style_tensor = self.graph[self.style_layers[layer_idx]]

        a_S = session.run(style_tensor)
        m, n_H, n_W, n_C = style_tensor.get_shape().as_list()
        a_G = style_tensor

        gshape = (m, n_H, n_W, n_C)

        GS = self._gram_matrix(a_S, gshape)
        GG = self._gram_matrix(a_G, gshape)

        J_style_layer = 1. / (4. * n_C * n_C * n_H * n_H * n_W * n_W) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))

        return J_style_layer

    def _style_loss(self):
        J_style = 0.0
        length = len(self.style_layers)
        coeff = 1.0 / length

        for i in range(length):
            J_style += self.style_layer_w[i] * self._single_style_loss(layer_idx=i)

        return J_style

    def total_loss(self, J_content, J_Style):
        self.J = self.content_w * J_content + self.style_w * J_Style

    def optimize(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.J, global_step=self.gstep)

    def build(self):
        self.load_vgg()
        J_content = self._content_loss()
        J_Style = self._style_loss()
        self.total_loss(J_content, J_Style)
        self.optimize()

    def train(self, epochs=300):
        # Initialize global variables
        session.run(tf.global_variables_initializer())
        session.run(self.graph['input'].assign(self.init_img))

        for i in range(epochs):
            session.run(self.optimizer)

            generated_img = session.run(self.graph['input'])

            # print every 20 iteration
            if i % 20 == 0:
                Jt = session.run(self.J)
                print("Iteration " + str(i) + " :")
                print("total cost = " + str(Jt))

                self.save_image("output/" + str(i) + ".png", generated_img)

        return generated_img

if __name__ == '__main__':
    session = tf.Session()
    content_img = "images/marvel.jpg"
    style_img = "images/harlequin.jpg"
    # 指定像素尺寸
    img_width = 400
    img_height = 300
    # style transfer
    style_transfer = StyleTransfer(content_img, style_img, img_width, img_height)

    style_transfer.build()
    generated_img = style_transfer.train(epochs=300)