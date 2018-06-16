import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import math

from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imread, imresize, imsave

import os

graph = tf.Graph()
with graph.as_default():
    inception5h_graph_def_file = os.path.join('inception/5h', 'tensorflow_inception_graph.pb')
    with tf.gfile.FastGFile(inception5h_graph_def_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

# tensor_names = [op.name for op in graph.get_operations()]
layer_names = ['conv2d0', 'conv2d1', 'conv2d2',
               'mixed3a', 'mixed3b',
               'mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e',
               'mixed5a', 'mixed5b']

tensor_name_input_image = "input:0"
input = graph.get_tensor_by_name(tensor_name_input_image)

# original: PIL.Image.open
# def load_image(filename):
#    image = PIL.Image.open(filename)
#    return np.float32(image)

# rewrite: scipy.misc
def load_image(filename):
    image = imread(filename)

    return np.float32(image)

def save_image(image, filename):
    image = np.clip(image, 0.0, 255.0)

    with open(filename, 'wb') as file:
        imsave(file, image)

def plot_image(image):
    image = np.clip(image / 255.0, 0.0, 1.0)

    plt.imshow(image, interpolation='lanczos')
    plt.show()

def normalize_image(x):
    x_min = x.min()
    x_max = x.max()

    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm

def plot_gradient(gradient):
    gradient_normalized = normalize_image(gradient)

    plt.imshow(gradient_normalized, interpolation='bilinear')
    plt.show()

def resize_image(image, size=None, factor=None):
    # assert factor is not None or size is not None

    if factor is not None:
        size = np.array(image.shape[0:2]) * factor
        size = size.astype(int)
    else:
        size = size[0:2]

    img_resized = imresize(image, size=size)
    return np.float32(img_resized)

# filename = "images/hulk.jpg"
# image = load_image(filename)
# print(image.shape)

# resized_img = resize_image(image, factor=0.7)
# print(resized_img.shape)
# save_image(resized_img, "images/hulk_save.jpg")

# DeepDream 算法
def get_tile_size(num_pixels, tile_size=400):
    """
    num_pixels is the number of pixels in a dimension of the image.
    tile_size is the desired tile-size.
    """

    # How many times can we repeat a tile of the desired size.
    num_tiles = int(round(num_pixels / tile_size))

    # Ensure that there is at least 1 tile.
    num_tiles = max(1, num_tiles)

    # The actual tile-size.
    actual_tile_size = math.ceil(num_pixels / num_tiles)        # math.ceil() 天花板除

    return actual_tile_size

def tiled_gradient(gradient, image, tile_size=400):
    grad = np.zeros_like(image)

    x_max, y_max, _ = image.shape
    x_tile_size = get_tile_size(num_pixels=x_max, tile_size=tile_size)
    x_tile_size4 = x_tile_size // 4

    y_tile_size = get_tile_size(num_pixels=y_max, tile_size=tile_size)
    y_tile_size4 = y_tile_size // 4

    x_start = random.randint(-3*x_tile_size4, -x_tile_size4)

    while x_start < x_max:
        x_end = x_start + x_tile_size

        x_start_lim = max(x_start, 0)
        x_end_lim = min(x_end, x_max)

        y_start = random.randint(-3*y_tile_size4, -y_tile_size4)

        while y_start < y_max:
            y_end = y_start + y_tile_size

            y_start_lim = max(y_start, 0)
            y_end_lim = min(y_end, y_max)

            img_tile = image[x_start_lim:x_end_lim, y_start_lim:y_end_lim, :]
            img_tile = np.expand_dims(img_tile, axis=0)

            feed_dict = {tensor_name_input_image: img_tile}
            g = session.run(gradient, feed_dict=feed_dict)
            g /= (np.std(g) + 1e-8)

            grad[x_start_lim:x_end_lim, y_start_lim:y_end_lim, :] = g

            y_start = y_end

        x_start = x_end

    return grad

# 优化
def optimize_image(layer_tensor, image, num_iterations=10, step_size=3.0, tile_size=400, show_gradient=False):
    img = image.copy()

    print("Image before:")
    plot_image(img)

    print("Processing image:", end="")

    with graph.as_default():
        tensor = tf.square(layer_tensor)
        tensor_mean = tf.reduce_mean(tensor)

        gradient = tf.gradients(tensor_mean, input)[0]

    for i in range(num_iterations):
        grad = tiled_gradient(gradient=gradient, image=img, tile_size=tile_size)

        sigma = (i * 4.0) / num_iterations + 0.5
        grad_smooth1 = gaussian_filter(grad, sigma=sigma)
        grad_smooth2 = gaussian_filter(grad, sigma=sigma)
        grad_smooth3 = gaussian_filter(grad, sigma=sigma)
        grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)

        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        img += step_size_scaled * grad

        if show_gradient:
            msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
            print(msg.format(grad.min(), grad.max(), step_size_scaled))
            # plot_gradient(grad)
        else:
            print(". ", end="")
    print()
    print("Image after:")
    plot_image(img)

    return img

# 递归优化
def recursive_optimize(layer_tensor, image, num_repeats=4, rescale_factor=0.7, blend=0.2, num_iterations=10, step_size=3.0, tile_size=400):

    if num_repeats > 0:
        sigma = 0.5
        img_blur = gaussian_filter(image, sigma=(sigma, sigma, 0.0))

        img_downscaled = resize_image(image=img_blur, factor=rescale_factor)

        result_image = recursive_optimize(layer_tensor, img_downscaled, num_repeats=num_repeats - 1, rescale_factor=rescale_factor,
                                          blend=blend, num_iterations=num_iterations, step_size=step_size, tile_size=tile_size)

        img_upscaled = resize_image(image=result_image, size=image.shape)

        image = blend * image + (1 - blend) * img_upscaled

    print("Recursive level:", num_repeats)

    result_image = optimize_image(layer_tensor=layer_tensor, image=image, num_iterations=num_iterations,
                                  step_size=step_size, tile_size=tile_size)

    return result_image

session = tf.Session(graph=graph)

image = load_image(filename='images/hulk.jpg')     # giger.jpg, escher_planefilling2.jpg

layer_tensor = graph.get_tensor_by_name(layer_names[2] + ":0")

# 单步优化
# img_result_one_step = optimize_image(layer_tensor, image,
#                    num_iterations=10, step_size=6.0, tile_size=400,
#                    show_gradient=True)

img_result_recursive_step = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=10, step_size=3.0, rescale_factor=0.7,
                 num_repeats=4, blend=0.2)

session.close()