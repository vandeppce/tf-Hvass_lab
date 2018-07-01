##############################################################
# 输入两张图像到神经网络中：一张内容图像和一张风格图像。
# 我们希望创建一张混合图像，它包含了内容图的轮廓以及风格图的纹理。
# 我们通过创建几个可以被优化的损失函数来完成这一点。

# 内容图像的损失函数会试着在网络的某一层或多层上，最小化内容图像以及
# 混合图像激活特征的差距。这使得混合图像和内容图像的的轮廓相似。

# 风格图像的损失函数稍微复杂一些，因为它试图让风格图像和混合图像的
# Gram矩阵的差异最小化。这在网络的一个或多个层中完成。 Gram-matrices
# 度量了哪个特征在给定层中同时被激活。改变混合图像，使其模仿风格图像的
# 激活模式(activation patterns)，这将导致颜色和纹理的迁移。
##############################################################
# from IPython.display import Image, display
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import PIL.Image

import inception5h

# model = inception.Inception()

def load_image(filename, max_size=None):
    image = PIL.Image.open(filename)

    if max_size is not None:
        # Calculate the appropriate rescale-factor for
        # ensuring a max height and width, while keeping
        # the proportion between them.
        factor = max_size / np.max(image.size)

        # Scale the image's height and width
        size = np.array(image.size) * factor

        # The size is now floating-point because it was scaled.
        # But PIL requires the size to be integers.
        size = size.astype(int)

        # Resize the image.
        image = image.resize(size, PIL.Image.LANCZOS)

    # Convert to numpy floating-point array
    return np.float32(image)

def save_image(image, filename):
    image = np.clip(image, 0.0, 255.0)

    image = image.astype(np.uint8)

    with open(filename, 'wb') as f:
        PIL.Image.fromarray(image).save(f, 'jpeg')

def plot_image_big(image):
    image = np.clip(image, 0.0, 255.0)

    image = image.astype(np.uint8)

    display(PIL.Image.fromarray(image))

# 绘制内容图像、混合图像以及风格图像
def plot_images(content_image, style_image, mixed_image):
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    smooth = True

    if smooth:
        interpolation = 'sinc'
    else:
        interpolation = 'nearest'

    ax = axes.flat[0]
    ax.imshow(content_image / 255.0, interpolation=interpolation)
    ax.set_xlabel('Content')

    ax = axes.flat[1]
    ax.imshow(mixed_image / 255.0, interpolation=interpolation)
    ax.set_xlabel('Mixed')

    ax = axes.flat[2]
    ax.imshow(style_image / 255.0, interpolation=interpolation)
    ax.set_xlabel('Style')

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

# 损失函数

# 计算输入tensor的最小平均误差(MSE)
def mean_squared_error(a, b):
    return tf.reduce_mean(tf.square(a - b))

# 内容损失函数。
# 给定层中，内容图像和混合图像激活特征的最小平均误差。当内容损失最小，
# 意味着在给定层中，混合图像和内容图像的激活特征很相似。

def create_content_loss(session, model, content_image, layer_ids):
    """
    Create the loss-function for the content-image.
    Parameters:
    :param session: An open TensorFLow session for running the model's graph.
    :param model: The model
    :param content_image: Numpy float array with the content-image
    :param layer_ids: List of integer id's for the layers to use in the model.
    :return:
    """

    # Create a feed-dict with the content-image
    feed_dict = model.create_feed_dict(image=content_image)

    layers = model.get_layer_tensors(layer_ids)

    # Calculate the output values of those layers when
    # feeding the content-image to the model.

    values = session.run(layers, feed_dict=feed_dict)

    # Set the model's graph as the default so we can add
    # computational nodes to it. It is not always clear
    # when this is necessary in TensorFlow, but if you
    # want to re-use this code then it may be necessary.

    with model.graph.as_default():
        layer_losses = []

        # For each layer and its corresponding values
        # for the content-image
        for value, layer in zip(values, layers):   # zip() 打包成元组
            # These are the values that are calculated
            # for this layer in the model when inputting
            # the content-image. Wrap it to ensure it
            # is a const - although this may be done
            # automatically by TensorFlow.
            value_cost = tf.constant(value)

            # The loss-function for this layer is the
            # Mean Squared Error between the layer-values
            # when inputting the content- and mixed-images.
            # Note that the mixed-image is not calculated
            # yet, we are merely creating the operations
            # for calculating the MSE between those two.

            loss = mean_squared_error(layer, value_cost)

            # Add the loss-function for this layer to the
            # list of loss-function
            layer_losses.append(loss)

        total_loss = tf.reduce_mean(layer_losses)

    return total_loss

# 对风格层做相同的处理，但需要度量出哪些特征在风格层中和风格图像中同时激活
# 一种方法是为风格层但输出张量计算一个Gram矩阵，Gram矩阵本质上就是风格层中激活特征向量的点乘矩阵
# 如果Gram矩阵中的一个元素的值接近于0，这意味着给定层的两个特征在风格图像中没有同时激活。
# 反之亦然，如果Gram矩阵中有很大的值，代表着两个特征同时被激活。

def gram_matrix(tensor):
    shape = tensor.get_shape()

    # Get the number of feature channels for the input tensor,
    # which is assumed to be from a convolutional layer with 4-dim
    num_channels = int(shape[3])

    # Reshape the tensor so it is a 2-dim matrix. This essentially
    # flattens the contents of each feature-channel.
    matrix = tf.reshape(tensor, shape=(-1, num_channels))

    # Calculate the Gram-matrix as the matrix-product of
    # the 2-dim matrix with itself. This calculates the
    # dot-products of all combinations of the feature-channels.
    gram = tf.matmul(tf.transpose(matrix), matrix)

    return gram

# 风格损失函数，计算Gram矩阵而非layer输出tensor的MSE

def create_style_loss(session, model, style_image, layer_ids):
    """
    Create the loss-function for the style-image.

    Parameters:
    :param session: An open TensorFlow session for running the model's graph.
    :param model: The model, e.g. an instance of the VGG16-model
    :param style_image: Numpy float array with the style-image.
    :param layer_ids: List of integers id's for the layers to use in the model.
    :return:
    """

    # Create a feed-dict with the style-image.
    feed_dict = model.create_feed_dict(image=style_image)

    # Get references to the tensors for the given layers.
    layers = model.get_layer_tensors(layer_ids)

    with model.graph.as_default():
        # Construct the TensorFlow-operations for calculating
        # the Gram-matrices for each of the layers.
        gram_layers = [gram_matrix(layer) for layer in layers]

        # Calculate the values of those Gram-matrices when
        # feeding the style-image to the model.
        values = session.run(gram_layers, feed_dict=feed_dict)

        # Initialize an empty list of loss-functions.
        layer_losses = []

        # For each Gram-matrix layer and its corresponding values.
        for value, gram_layer in zip(values, gram_layers):
            value_const = tf.constant(value)

            # The loss-function for this layer is the
            # Mean Squared Error between the Gram-matrix values
            # for the content- and mixed-images.
            # Note that the mixed-image is not calculated
            # yet, we are merely creating the operations for
            # calculating the MSE between those two.
            loss = mean_squared_error(gram_layer, value_const)

            # Add the loss-function for this layer to the
            # list of loss-functions.
            layer_losses.append(loss)

        total_loss = tf.reduce_mean(layer_losses)

    return total_loss

# 给混合图像去噪。算法是Total Variation Denoising, 本质上就是在x轴和y轴上
# 将图像偏移一个像素，计算它与原始图像的差异，取绝对值保证差异是正值，然后对
# 整个图像的所有像素求和。这个步骤创建了一个可以最小化的损失函数，用来
# 抑制图像中的噪声。

def create_denoise_loss(model):
    loss = tf.reduce_sum(tf.abs(model.input[:,1:,:,:] - model.input[:,:-1,:,:])) + \
        tf.reduce_sum(tf.abs(model.input[:, :, 1:, :] - model.input[:, :, :-1, :]))

    return loss

# 风格迁移
# 算法使用了损失函数归一化，在每次优化迭代中，调整损失值，使它们等于一。这可以让
# 用户独立地设置所选风格层以及内容层的损失权重。同时，在优化过程中也修改权重，
# 来确保保留风格、内容、去噪之间所需的比重。

def style_transfer(content_image, style_image, content_layer_ids, style_layer_ids,
                   weight_content=1.5, weight_style=10.0,
                   weight_denoise=0.3,
                   num_iterations=120, step_size=10.0):
    """
    Use gradient descent to find an image that minimizes the
    loss-functions of the content-layers and style-layers. This
    should result in a mixed-image that resembles the contours
    of the content-image, and resembles the colours and textures
    of the style-image.

    :param content_image: Numpy 3-dim float-array with the content-image.
    :param style_image: Numpy 3-dim float-array with the style-image.
    :param content_layer_ids: List of integers identifying the content-layers.
    :param style_layer_ids: List of integers identifying the style-layers.
    :param weight_content: Weight of the content-loss-function.
    :param weight_style: Weight of the style-loss-function.
    :param weight_denoise: Weight of the denoising-loss-function.
    :param num_iterations: Number of optimization iterations to perform.
    :param step_size: Step-size for the gradient in each iteration.
    :return:
    """

    # Create an instance of the Inception-model. This is done
    # in each call of this function, because we will add
    # operations to the graph so it can grow very large
    # and run out of RAM if we keep using the same instance.
    model = inception5h.Inception5h()

    # Create a TensorFlow-session.
    session = tf.InteractiveSession(graph=model.graph)

    # Print the names of the content-layers.
    print("Content layers:")
    print(model.get_layer_names(content_layer_ids))
    print()

    # Print the names of the style-layers.
    print("Style layers:")
    print(model.get_layer_names(style_layer_ids))
    print()

    # Create the loss-function for the content-layers and -image.
    loss_content = create_content_loss(session=session,
                                       model=model,
                                       content_image=content_image,
                                       layer_ids=content_layer_ids)

    # Create the loss-function for the style-layers and -image.
    loss_style = create_style_loss(session=session,
                                   model=model,
                                   style_image=style_image,
                                   layer_ids=style_layer_ids)

    # Create the loss-function for the denoising of the mixed-image.
    loss_denoise = create_denoise_loss(model)

    # Create TensorFlow variables for adjusting the values of
    # the loss-functions. This is explained below.
    adj_content = tf.Variable(1e-10, name='adj_content')
    adj_style = tf.Variable(1e-10, name='adj_style')
    adj_denoise = tf.Variable(1e-10, name='adj_denoise')

    # Initialize the adjustment values for the loss-function.
    session.run([adj_content.initializer,
                 adj_style.initializer,
                 adj_denoise.initializer])

    # Create TensorFlow operations for updating the adjustment values.
    # These are basically just the reciprocal values of the
    # loss-functions, with a small value 1e-10 added to avoid the
    # possibility of division by zero.

    update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
    update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))
    update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))

    # This is the weighted loss-function that we will minimize
    # below in order to generate the mixed-image.
    # Because we multiply the loss-values with their reciprocal adjustment values,
    # we can use relative weights for the loss-functions that are easier
    # to select, as they are independent of the exact choice of style- and content-layers.
    loss_combined = weight_content * adj_content * loss_content + \
                    weight_style * adj_style * loss_style + \
                    weight_denoise * adj_denoise * loss_denoise

    # Use TensorFlow to get the mathematical function for the
    # gradient of the combined loss-function with regard to
    # the input image.
    gradient = tf.gradients(loss_combined, model.input)

    # List of tensors that we will run in each optimization iteration.
    run_list = [gradient, update_adj_content, update_adj_style, update_adj_denoise]

    # The mixed-image is initialized with random noise.
    # It is the same size as the content-image.
    mixed_image = np.random.rand(*content_image.shape) + 128

    for i in range(num_iterations):
        feed_dict = model.create_feed_dict(image=mixed_image)

        # Use TensorFlow to calculate the value of the
        # gradient, as well as updating the adjustment values.
        grad, adj_content_val, adj_style_val, adj_denoise_val \
            = session.run(run_list, feed_dict=feed_dict)

        # Reduce the dimensionality of the gradient.
        grad = np.squeeze(grad)

        # Scale the step-size according to the gradient-values.
        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        # Update the image by following the gradient.
        mixed_image -= grad * step_size_scaled

        # Ensure the image has valid pixel-values between 0 and 255.
        mixed_image = np.clip(mixed_image, 0.0, 255.0)

        # Print a little progress-indicator.
        print(". ", end="")

        # Display status once every 10 iterations, and the last.
        if (i % 100 == 0) or (i == num_iterations - 1):
            print()
            print("Iteration:", i)

            # Print adjustment weights for loss-functions.
            msg = "Weight Adj. for Content: {0:.2e}, Style: {1:.2e}, Denoise: {2:.2e}"
            print(msg.format(adj_content_val, adj_style_val, adj_denoise_val))

            # Plot the content-, style- and mixed-images.
            plot_images(content_image=content_image,
                        style_image=style_image,
                        mixed_image=mixed_image)

    print()
    print("Final image:")
    plot_image_big(mixed_image)

    # Close the TensorFlow session to release its resources.
    session.close()

    return mixed_image

content_filename = 'images/willy_wonka_old.jpg'
content_image = load_image(content_filename, max_size=None)

style_filename = 'images/style7.jpg'
style_image = load_image(style_filename, max_size=300)

content_layer_ids = [2, 3, 4, 5, 6, 7, 8]
style_layer_ids = [4]

img = style_transfer(content_image=content_image,
                     style_image=style_image,
                     content_layer_ids=content_layer_ids,
                     style_layer_ids=style_layer_ids,
                     weight_content=5,
                     weight_style=2.0,
                     weight_denoise=0.3,
                     num_iterations=1000,
                     step_size=10.0)

