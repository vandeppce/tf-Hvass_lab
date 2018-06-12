import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

import inception

# inception.maybe_download()
model = inception.Inception()

# 取得Inception模型输入张量的引用。这个张量是用来保存调整大小后的图像，
# 即299 x 299像素并带有3个颜色通道。我们会在调整大小后的图像上添加噪声，
# 然后还是用这个张量将结果传到图（graph）中，因此需要确保调整大小的算法没有引入噪声。

resized_image = model.resized_image

# 获取Inception模型softmax分类器输出的引用。

y_pred = model.y_pred

# 获取Inception模型softmax分类器未经尺度变化的（unscaled）输出的引用。
# 这通常称为“logits”。由于我们会在graph上添加一个新的损失函数，其中用到这些未经变化的输出，
# 因此logits是必要的。

y_logits = model.y_logits

# Set the graph for the Inception model as the default graph,
# so that all changes inside this with-block are done to that graph.

with model.graph.as_default():
    # Add a placeholder variable for the target class-number.
    # This will be set to e.g. 300 for the 'bookcase' class.
    pl_cls_target = tf.placeholder(tf.int32)

    # Add a new loss-function. This is the cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[pl_cls_target], logits=y_logits)

    # Get the gradient for the loss function with regard to the resized input image.
    gradient = tf.gradients(ys=loss, xs=resized_image)


session = tf.Session(graph=model.graph)

# 寻找对抗噪声
# 本质上用梯度下降执行优化，噪声初始化为零，然后用损失函数关于输入噪声图像的梯度逐步优化

def find_adversary_noise(image_path, cls_target, noise_limit=3.0, required_score=0.99, max_iterations=100):
    """
    Find the noise that must be added to the given image so that it is classified as the target-class.

    :param image_path: File-path to the input-image (must be *.jpg).
    :param cls_target: Target class-number (integer between 1-1000
    :param noise_limit: Limit for pixel-values in the noise.
    :param required_score: Stop when target-class score reaches this.
    :param max_iterations: Max number of optimization iterations to perform.

    """

    # Create a feed-dict with the image
    feed_dict = model._create_feed_dict(image_path=image_path)

    # Use Tensorflow the calculate the predicted class-scores
    pred, image = session.run([y_pred, resized_image], feed_dict=feed_dict)

    # Convert to one-dimensional array.
    pred = np.squeeze(pred)

    # Predicted class-number
    cls_source = np.argmax(pred)

    # Score for the predicted class
    score_source_org = pred.max()

    # Name for the source and target classes
    name_source = model.name_lookup.cls_to_name(cls_source,
                                                only_first_name=True)
    name_target = model.name_lookup.cls_to_name(cls_target,
                                                only_first_name=True)

    # Initialize the noise to zero
    noise = 0

    # iterations
    for i in range(max_iterations):
        print("Iterations: ", i)

        # The noisy image is just the sum of the input image and noise.
        noisy_image = image + noise

        # 0-255
        noisy_image = np.clip(a=noisy_image, a_min=0.0, a_max=255.0)

        # Create a feed-dict, this feeds the noisy image to the tensor in
        # the graph that holds the resized image, because this is the final
        # stage for inputting raw image data.
        # This also feeds the target class-number that we desire

        feed_dict = {model.tensor_name_resized_image: noisy_image, pl_cls_target: cls_target}

        pred, grad = session.run([y_pred, gradient], feed_dict=feed_dict)

        pred = np.squeeze(pred)

        score_source = pred[cls_source]
        score_target = pred[cls_target]

        grad = np.array(grad).squeeze()

        # Calculate the max of the absolute gradient values
        # This is used to calculate the step-size
        grad_absmax = np.abs(grad).max()

        if grad_absmax < 1e-10:
            grad_absmax = 1e-10

        step_size = 7 / grad_absmax

        msg = "Source score: {0:>7.2%}, class-number: {1:4}, class-name: {2}"
        print(msg.format(score_source, cls_source, name_source))

        msg = "Target score: {0:>7.2%}, class-number: {1:4}, class-name: {2}"
        print(msg.format(score_target, cls_target, name_target))

        msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
        print(msg.format(grad.min(), grad.max(), step_size))

        print()

        if score_target < required_score:
            noise -= step_size * grad

            noise = np.clip(a=noise,
                            a_min=-noise_limit,
                            a_max=noise_limit)
        else:
            break

    return image.squeeze(), noisy_image.squeeze(), noise, name_source, name_target, score_source, score_source_org, score_target

# 输入归一化
def normalize_image(x):
    x_min = x.min()
    x_max = x.max()

    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm

def plot_image(image, noise, noisy_image, name_source, name_target, score_source, score_source_org, score_target):

    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    smooth = True

    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    ax = axes.flat[0]
    ax.imshow(image / 255.0, interpolation=interpolation)
    msg = "Original Image: \n{0} ({1:.2%})"
    xlabel = msg.format(name_source, score_source_org)
    ax.set_xlabel(xlabel)

    ax = axes.flat[1]
    ax.imshow(noisy_image / 255.0, interpolation=interpolation)
    msg = "Image + Noise:\n{0} ({1:.2%})\n{2} ({3:.2%})"
    xlabel = msg.format(name_source, score_source, name_target, score_target)
    ax.set_xlabel(xlabel)

    ax = axes.flat[2]
    ax.imshow(normalize_image(noise), interpolation=interpolation)
    xlabel = "Amplified Noise"
    ax.set_xlabel(xlabel)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def adversary_example(image_path, cls_target, noise_limit, required_score):
    image, noisy_image, noise, \
    name_scoure, name_target, \
    score_source, score_source_org, score_target = \
        find_adversary_noise(image_path=image_path,
                             cls_target=cls_target,
                             noise_limit=noise_limit,
                             required_score=required_score)

    plot_image(image=image, noise=noise, noisy_image=noisy_image,
               name_source=name_scoure, name_target=name_target,
               score_source=score_source,
               score_source_org=score_source_org,
               score_target=score_target)

    msg = "Noise min: {0:.3f}, max: {1:.3f}, mean: {2:.3f}, std: {3:.3f}"
    print(msg.format(noise.min(), noise.max(),
                     noise.mean(), noise.std()))

image_path = "images/parrot_cropped1.jpg"
adversary_example(image_path=image_path,
                  cls_target=300,
                  noise_limit=1.0,
                  required_score=0.99)
