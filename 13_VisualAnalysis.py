#####################################################################
# 在之前的关于CNN的教程中，展示了滤波权重，
# 但是单从滤波权重上看，无法确定滤波器会从输入图像中识别出什么。
# 本教程中，会提出一种用于可视化分析神经网络内部工作原理的基本方法，
# 这个方法就是生成最大化神经网络内个体特征的图像。
# 图像用一些随机噪声初始化，然后用给定特征关于输入图像的梯度来逐渐改变生成图像。
# 这种方法也被称为"特征最大化"或者"激活最大化"
#####################################################################

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import inception

# 返回inception中的卷积层名称列表

def get_conv_layer_name():
    model = inception.Inception()
    names = [op.name for op in model.graph.get_operations() if op.type == 'Conv2D']

    model.close()

    return names

conv_names = get_conv_layer_name()

# print(len(conv_names))        # 94
# print(conv_names[:5])
# print(conv_names[-5:])

# 寻找使网络内给定特征最大化的输入图像，本质上采用梯度法进行优化。
# 图像用小的随机值初始化，然后用给定特征关于输入图像的梯度逐步更新。

def optimize_image(conv_id=None, feature=0, num_iterations=30, show_progress=True):
    """
    Find an image that maxmizes the feature given by the conv_id and feature number.

    Arguments:
    :param conv_id: Integer identifying the convolutional layer to maximize. It is an
                    index into conv_names. If None then use the last fully-connected
                    layer before the softmax output.
    :param feature: Index into the layer for the feature to maximize.
    :param num_iterations: Number of optimization iterations to perform.
    :param show_progress: Boolean whether to show the progress.

    :return:
    image
    """

    # Load the Inception model. This is done for each call of
    # this function because we will add a lot to the graph which
    # will cause the graph to grow and eventually the computer
    # will run out of memory.

    model = inception.Inception()

    # Reference to the tensor that takes the raw input image.
    resized_image = model.resized_image

    # Reference to the tensor for the predicted classes.
    # This is the output of the final layer's softmax classifier.
    y_pred = model.y_pred

    # Create the loss-function that must be maximized.
    if conv_id is None:
        # If we want to maximize a feature on the last layer,
        # then we use the fully-connected layer prior to the
        # softmax-classifier. The feature no. is the class-number
        # and must be an integer between 1 and 1000.
        # The loss-function is just the value of that feature.
        loss = model.y_logits[0, feature]
    else:
        # If instead we want to maximize a feature of a
        # convolutional layer inside the neural network.

        # Get the name of the convolutional operator.
        conv_name = conv_names[conv_id]

        # Get a reference to the tensor that is output by the
        # operator. Note that ":0" is added to the name for this.
        tensor = model.graph.get_tensor_by_name(conv_name + ":0")

        # Set the Inception model's graph as default
        # so we can add an operator to it.
        with model.graph.as_default():
            # The loss-funciton is the average of all the
            # tensor-values for the given feature. This
            # ensures that we generate the whole input image.
            # You can try and modify this so it only uses
            # a part of the tensor.
            loss = tf.reduce_mean(tensor[:, :, :, feature])

    # Get the gradient for the loss-function which regard to
    # the resized input image. This creates a mathematical
    # function for calculating the gradient.
    gradient = tf.gradients(ys=loss, xs=resized_image)

    session = tf.Session(graph=model.graph)

    # Generate a random image of the same size as the raw input.
    # Each pixel is a small random value between 128 and 129,
    # which is about the middle of the colour-range.
    image_shape = resized_image.get_shape()
    image = np.random.uniform(size=image_shape) + 128.0

    # Perform a number of optimization iterations to find
    # the image that maximizes the loss-function.
    for i in range(num_iterations):
        # Create a feed-dict. This feeds the image to the
        # tensor in the graph that holds the resized image, because
        # this is the final stage for inputting raw image data.
        feed_dict = {model.tensor_name_resized_image: image}

        pred, grad, loss_value = session.run([y_pred, gradient, loss],
                                             feed_dict=feed_dict)

        # Squeeze the dimensionality for the gradient-array.
        grad = np.array(grad).squeeze()

        # The gradient now tells us how much we need to change the
        # input image in order to maximize the given feature.

        # Calculate the step-size for updating the image.
        # This step-size was found to give fast convergence.
        # The addition of 1e-8 is to protect from div-by-zero.
        step_size = 1.0 / (grad.std() + 1e-8)

        # Update the image by adding the scaled gradient
        # This is called gradient ascent. 梯度上升
        image += step_size * grad

        # Ensure all pixel-values in the image are between 0 and 255.
        image = np.clip(image, 0.0, 255.0)

        if show_progress:
            print("Iteration:", i)

            # Convert the predicted class-scores to a one-dim array.
            pred = np.squeeze(pred)

            # The predicted class for the Inception model.
            pred_cls = np.argmax(pred)

            # Name of the predicted class.
            cls_name = model.name_lookup.cls_to_name(pred_cls,
                                                     only_first_name=True)

            # The score (probability) for the predicted class.
            cls_score = pred[pred_cls]

            # Print the predicted score etc.
            msg = "Predicted class-name: {0} (#{1}), score: {2:>7.2%}"
            print(msg.format(cls_name, pred_cls, cls_score))

            # Print statistics for the gradient.
            msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
            print(msg.format(grad.min(), grad.max(), step_size))

            # Print the loss-value.
            print("Loss:", loss_value)

            # Newline.
            print()

            # Close the TensorFlow session inside the model-object.
    model.close()
    session.close()
    return image.squeeze()

# 绘制

def normalize_image(x):
    x_min = x.min()
    x_max = x.max()

    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm

# 画一张图
def plot_image(image):
    # Normalize the image so pixels are between 0.0 and 1.0
    img_norm = normalize_image(image)

    # Plot the image.
    plt.imshow(img_norm, interpolation='nearest')
    plt.show()

# 画六张图
def plot_images(images, show_size=100):
    """
    The show_size is the number of pixels to show for each image.
    The max value is 299.
    """

    fig, axes = plt.subplots(2, 3)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Use interpolation to smooth pixels?
    smooth = True

    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        img = images[i, 0:show_size, 0:show_size, :]
        img_norm = normalize_image(img)

        ax.imshow(img_norm, interpolation=interpolation)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

# 优化多张图片并绘制
def optimize_images(conv_id=None, num_iterations=30, show_size=100):
    """
    Find 6 images that maximize the 6 first features in the layer
    given by the conv_id.

    Parameters:
    conv_id: Integer identifying the convolutional layer to
             maximize. It is an index into conv_names.
             If None then use the last layer before the softmax output.
    num_iterations: Number of optimization iterations to perform.
    show_size: Number of pixels to show for each image. Max 299.
    """

    if conv_id is None:
        print("Final fully-connected layer before softmax.")
    else:
        print("Layer: ", conv_names[conv_id])

    images = []

    # For each feature do the following. Note that the
    # last fully-connected layer only supports numbers
    # between 1 and 1000, while the convolutional layers
    # support numbers between 0 and some other number.
    # So we just use the numbers between 1 and 7.

    for feature in range(1, 7):
        print("Optimizing image for feature no.", feature)

        # Fine the image that maximizes the given feature
        # for the network layer identified by conv_id (or None)
        image = optimize_image(conv_id=conv_id, feature=feature,
                               show_progress=False,
                               num_iterations=num_iterations)

        # Squeeze the dim of the array
        image = image.squeeze()

        images.append(image)

    images = np.array(images)

    plot_images(images=images, show_size=show_size)

# 结果：单个特征，特征即滤波器
image = optimize_image(conv_id=5, feature=2, num_iterations=30, show_progress=True)
plot_image(image)

# 结果：多张图像，更深的层图案更复杂

optimize_images(conv_id=3, num_iterations=30)
optimize_images(conv_id=4, num_iterations=30)
optimize_images(conv_id=10, num_iterations=30)
optimize_images(conv_id=20, num_iterations=30)
optimize_images(conv_id=30, num_iterations=30)
optimize_images(conv_id=60, num_iterations=30)
optimize_images(conv_id=90, num_iterations=30)
optimize_images(conv_id=93, num_iterations=30)
optimize_images(conv_id=None, num_iterations=30)