#################################################################################
# 教程11和12展示了如何使用梯度生成对抗噪声，
# 教程13展示了怎么用梯度生成神经网络内部特征响应的图像。
# 本文使用梯度来放大输入图像中的图案，称为DeepDream算法。
# 算法使用TensorFlow自动导出网络中一个给定层相对于输入图像的梯度。
# 然后用梯度来更新输入图像。这个过程重复多次，直到出现图案并且我们对所得到的图像满意为止。
# 这里的原理就是，神经网络在图像中看到一些图案的痕迹，然后我们只是用梯度把它放大了。
# 这里没有显示DeepDream算法的一些细节，例如梯度被平滑了，后面会讨论它的一些优点。
# 梯度也是分块计算的，因此它可以在高分辨率的图像上工作，而不会耗尽计算机内存。

# 递归优化
# Inception模型是在相当低分辨率的图像上进行训练的，大概200-300像素。
# 所以，当我们使用更大分辨率的图像时，DeepDream算法会在图像中创建许多小的图案。
# 一个解决方案是将输入图像缩小到200-300像素。但是这么低的分辨率（的结果）是像素化而且丑陋的。
# 另一个解决方案是多次缩小原始图像，在每个较小的图像上运行DeepDream算法。
# 这样会在图像中创建更大的图案，然后以更高的分辨率进行改善。
#################################################################################

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import math

import PIL.Image
from scipy.ndimage.filters import gaussian_filter

# 这里使用inception5h模型，它接受任何尺寸的输入图像，
# 然后创建比Inception v3模型更漂亮的图像。

import inception5h
model = inception5h.Inception5h()

# layer_names = [op.name for op in model.graph.get_operations() if op.type == 'Conv2D']
# print(layer_names[0])
# print(len(model.layer_tensors))     # 12

# 操作图像
def load_image(filename):
    image = PIL.Image.open(filename)

    return np.float32(image)

def save_image(image, filename):
    image = np.clip(image, 0.0, 255.0)

    # Convert to bytes
    image = image.astype(np.uint8)

    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')

# 绘制图像，使用matplotlib得到低分辨率图像，使用PIL效果比较好
def plot_image(image):
    if True:
        image = np.clip(image/255.0, 0.0, 1.0)

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

# 这个函数调整图像的大小。函数的参数是你指定的具体的图像分辨率，比如(100，200)，
# 它也可以接受一个缩放因子，比如，参数是0.5时,图像每个维度缩小一半。

# 这个函数用PIL来实现，代码有点长，因为我们用numpy数组来处理图像，其中像素值是浮点值。
# PIL不支持这个，因此需要将图像转换成8位字节，来确保像素值在合适的范围内。
# 然后，图像被调整大小并转换回浮点值。

def resize_image(image, size=None, factor=None):
    # If a rescaling-factor is provided then use it.
    if factor is not None:
        # Scale the numpy array's shape for height and width
        size = np.array(image.shape[0:2]) * factor

        # The size is floating-point because it was scaled.
        # PIL requires the size to be integers.
        size = size.astype(int)
    else:
        # Ensure the size has length 2
        size = size[0:2]

    # The height and width is reversed in numpy vs. PIL
    size = tuple(reversed(size))

    # Ensure the pixel-values are between 0 and 255.
    img = np.clip(image, 0.0, 255.0)

    # Convert the pixels to 8-bit bytes
    img = img.astype(np.uint8)

    # Create PIL-object from numpy array.
    img = PIL.Image.fromarray(img)

    # Resize the image.
    img_resized = img.resize(size, PIL.Image.LANCZOS)

    # Convert 8-bit pixel values back to floating-point.
    img_resized = np.float32(img_resized)

    return img_resized

# DeepDream 算法

# Inception 5h模型可以接受任意尺寸的图像，但太大的图像可能会占用千兆字节的内存。
# 为了使内存占用最低，我们将输入图像分割成小的图块，然后计算每小块的梯度。
# 然而，这可能会在DeepDream算法最终生成的图像中产生肉眼可见的线条。
# 因此我们随机地挑选小块，这样它们的位置就是不同的。这使得在最终的DeepDream图像里，小块之间的缝隙不可见。

def get_tile_size(num_pixels, tile_size=400):
    """

    :param num_pixels: number of pixels in a dimension of the image
    :param tile_size: desired tile-size
    :return:
    """

    # How many times can we repeat a tile of the desired size.
    num_tiles = int(round(num_pixels / tile_size))

    # Ensure that there is at least 1 tile
    num_tiles = max(1, num_tiles)

    # The actual tile-size
    actual_tile_size = math.ceil(num_pixels / num_tiles)

    return actual_tile_size

# 计算输入图像的梯度。图像被分割成小块，然后分别计算各个图块的梯度，
# 图块随机选择，避免在最终的DeepDream图像内产生可见的缝隙

def tiled_gradient(gradient, image, tile_size=400):
    grad = np.zeros_like(image)

    x_max, y_max, _ = image.shape

    # Tile-size for the x-axis
    x_tile_size = get_tile_size(num_pixels=x_max, tile_size=tile_size)

    # 1/4 of the tile-size
    x_tile_size4 = x_tile_size // 4

    # Tile-size for the y-axis
    y_tile_size = get_tile_size(num_pixels=y_max, tile_size=tile_size)

    # 1/4 of the tile-size
    y_tile_size4 = y_tile_size // 4

    # Random start-position for the tiles on the x-axis.
    # The random value is between -3/4 and -1/4 of the tile-size.
    # This is so the border-tiles are at least 1/4 of the tile-size,
    # otherwises the tiles may be too small which creates noisy gradients.

    x_start = random.randint(-3*x_tile_size4, -x_tile_size4)

    while x_start < x_max:
        # End-position for the current tile.
        x_end = x_start + x_tile_size

        # Ensure the tile's start and end positions are valid.
        x_start_lim = max(x_start, 0)
        x_end_lim = min(x_end, x_max)

        # Random start-position for the tiles on the y-axis.
        # The random value is between -3/4 and -1/4 of the tile-size
        y_start = random.randint(-3*y_tile_size4, -y_tile_size4)

        while y_start < y_max:
            y_end = y_start + y_tile_size

            y_start_lim = max(y_start, 0)
            y_end_lim = min(y_end, y_max)

            # Get the image tile
            img_tile = image[x_start_lim:x_end_lim, y_start_lim:y_end_lim, :]

            # Create a feed-dict with the image-tile
            feed_dict = model.create_feed_dict(image=img_tile)

            # Use TensorFlow to calculate the gradient-value
            g = session.run(gradient, feed_dict=feed_dict)

            # Normalize the gradient for the tile. This is
            # necessary because the tiles may have very different
            # values. Normalizing gives a more coherent gradient.
            g /= (np.std(g) + 1e-8)

            # Store the tile's gradient at the appropriate location.
            grad[x_start_lim:x_end_lim,
            y_start_lim:y_end_lim, :] = g

            # Advance the start-position for the y-axis
            y_start = y_end

        # Advance the start-position for the x-axis
        x_start = x_end
    return grad

# 优化图像
# 这个函数是DeepDream算法的主要优化循环。它根据输入图像计算Inception模型中给定层的梯度。
# 然后将梯度添加到输入图像，从而增加层张量(layer-tensor)的平均值。
# 多次重复这个过程，并放大Inception模型在输入图像中看到的任何图案。

def optimize_image(layer_tensor, image, num_iterations=10, step_size=3.0, tile_size=400, show_gradient=False):
    """
    Use gradient ascent to optimize an image so it maximizes the mean value of the given layer_tensor

    :param layer_tensor: Reference to a tensor that will be maximized.
    :param image: Input image used as the starting point.
    :param num_iterations: Number of optimization iterations to perform.
    :param step_size: Scale for each step of the gradient ascent.
    :param tile_size: Size of the tiles when calculating the gradient.
    :param show_gradient: Plot the gradient in each iterations.

    :return:
    """

    # Copy the image so we don't overwrite the original image.
    img = image.copy()

    print("Image before:")
    plot_image(img)

    print("Processing image:", end="")

    # Use TensorFlow to get the mathematical function for the
    # gradient of the given layer-tensor with regard to the
    # input image. This may cause TensorFlow to add the same
    # math-expressions to the graph each time this function is called.
    # It may use a lot of RAM and could be moved outside the function.

    gradient = model.get_gradient(layer_tensor)

    for i in range(num_iterations):
        # Calculate the value of the gradient.
        # This tells us how to change the image so as to
        # maximize the mean of the given layer-tensor.
        grad = tiled_gradient(gradient=gradient, image=img)

        # Blur the gradient with different amounts and add
        # them together. The blur amount is also increased
        # during the optimization. This was found to give
        # nice, smooth images. You can try and change the formulas.
        # The blur-amount is called sigma (0=no blur, 1=low blur, etc.)
        # We could call gaussian_filter(grad, sigma=(sigma, sigma, 0.0))
        # which would not blur the colour-channel. This tends to
        # give psychadelic / pastel colours in the resulting images.
        # When the colour-channel is also blurred the colours of the
        # input image are mostly retained in the output image.

        sigma = (i * 4.0) / num_iterations + 0.5
        grad_smooth1 = gaussian_filter(grad, sigma=sigma)
        grad_smooth2 = gaussian_filter(grad, sigma=sigma * 2)
        grad_smooth3 = gaussian_filter(grad, sigma=sigma*0.5)
        grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)

        # Scale the step-size according to the gradient-values.
        # This may not be necessary because the tiled-gradient
        # is already normalized

        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        # Update the image by following the gradient.
        img += grad * step_size_scaled

        if show_gradient:
            # Print statistics for the gradient.
            msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
            print(msg.format(grad.min(), grad.max(), step_size_scaled))

            # Plot the gradient.
            plot_gradient(grad)
        else:
            # Otherwise show a little progress-indicator.
            print(". ", end="")

    print()
    print("Image after:")
    plot_image(img)

    return img

# 图像递归优化
# Inception模型在相当小的图像上进行训练。不清楚图像的确切大小，但可能每个维度200-300像素。
# 如果我们使用较大的图像，比如1920x1080像素，那么上面的optimize_image()函数会在图像上添加很多小的图案。

# 下面这个帮助函数将输入图像多次缩放，然后用每个缩放图像来执行上面的optimize_image()函数。
# 这在最终的图像中生成较大的图案。它也能加快计算速度。

def recursive_optimize(layer_tensor, image, num_repeats=4, rescale_facotr=0.7,
                       blend=0.2, num_iterations=10, step_size=3.0, tile_size=400):
    """
    Recursively blur and downscale the input image.
    Each downscaled image is run through the optimize_image()
    function to amplify the patterns that the Inception model sess.

    :param Input: Input image used as the starting point.
    :param num_repeats: Number of times to downscale the image.
    :param rescale_facotr: Downscaling factor for the image.
    :param blend: Factor for blending the original and processed images.

    Parameters passed to optimize_image()
    :param num_iterations: Number of optimization iterations to perform.
    :param step_size: Scale for each step of the gradient ascent.
    :param tile_size: Size of the tiles when calculating the gradient.
    :param layer_tensor: Reference to a tensor that will be maximized.

    :return:
    """

    # Do a recursive step?
    if num_repeats > 0:
        # Blur the input image to prevent artifacts when downscaling.
        # The blur amount is controlled by sigma. Note that the
        # colour-channel is not blurred as it would make the image grey.
        sigma = 0.5
        img_blur = gaussian_filter(image, sigma=(sigma, sigma, 0.0))

        # Downscale the image.
        img_downscaled = resize_image(image=img_blur, factor=rescale_facotr)

        # Recursive call to this function.
        # Subtract one from num_repeats and use the downscaled image.
        img_result = recursive_optimize(layer_tensor=layer_tensor, image=img_downscaled, num_repeats=num_repeats-1,
                                        rescale_facotr=rescale_facotr, blend=blend, num_iterations=num_iterations,
                                        step_size=step_size, tile_size=tile_size)

        # Up scale the resulting image back to its original size
        img_upscaled = resize_image(image=img_result, size=image.shape)

        # Blend the original and precessed images.
        image = blend * image + (1.0 - blend) * img_upscaled

    print("Recursive level:", num_repeats)

    img_result = optimize_image(layer_tensor=layer_tensor, image=image, num_iterations=num_iterations, step_size=step_size, tile_size=tile_size)

    return img_result

# 创建session，这是一个交互式的会话，可以继续朝图中添加梯度方程
session = tf.InteractiveSession(graph=model.graph)

image = load_image(filename='images/hulk.jpg')
plot_image(image)

'''
layer_tensor = model.layer_tensors[2]
img_result = optimize_image(layer_tensor, image,
                   num_iterations=10, step_size=6.0, tile_size=400,
                   show_gradient=True)

# 现在，递归调用DeepDream算法。我们执行5个递归（num_repeats + 1），
# 每个步骤中图像都被模糊并缩小，然后在缩小图像上运行DeepDream算法。
# 接着，在每个步骤中，将产生的DeepDream图像与原始图像混合，从原始图像获取一点细节。
# 这个过程重复了多次。

img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                                num_iterations=10, step_size=3.0, rescale_facotr=0.7,
                                num_repeats=4, blend=0.2)

# 现在我们将最大化Inception模型中的较高层。
# 使用7号层（索引6）为例。该层识别输入图像中更复杂的形状，
# 所以DeepDream算法也将产生更复杂的图像。
# 这一层似乎识别了狗的脸和毛发，因此DeepDream算法往图像中添加了这些东西。

# 再次注意，与DeepDream算法其他变体不同的是，这里输入图像的大部分颜色被保留了下来，
# 创建了更多柔和的颜色。这是因为我们在颜色通道中平滑了梯度，使其变得有点像灰阶，
# 因此不会太多地改变输入图像的颜色。

layer_tensor = model.layer_tensors[6]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                                num_iterations=10, step_size=3.0, rescale_facotr=0.7,
                                num_repeats=4, blend=0.2)
'''
layer_tensor = model.layer_tensors[11][:,:,:,0]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                                num_iterations=10, step_size=3.0, rescale_facotr=0.7,
                                num_repeats=4, blend=0.2)

save_image(img_result, 'images/hulk_deepdream.jpg')

image = load_image(filename='images/giger.jpg')
plot_image(image)

layer_tensor = model.layer_tensors[5]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                                num_iterations=10, step_size=3.0, rescale_facotr=0.7,
                                num_repeats=4, blend=0.2)

save_image(img_result, 'images/giger_deepdream.jpg')

image = load_image(filename='images/escher_planefilling2.jpg')
plot_image(image)

layer_tensor = model.layer_tensors[6]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                                num_iterations=10, step_size=3.0, rescale_facotr=0.7,
                                num_repeats=4, blend=0.2)

save_image(img_result, 'images/escher_deepdream.jpg')

session.close()