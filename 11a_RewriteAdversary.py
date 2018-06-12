import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

import inception

model = inception.Inception()

inception_graph_def_file = os.path.join('inception', 'classify_image_graph_def.pb')

with tf.gfile.FastGFile(inception_graph_def_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

tensor_name_input_jpeg = "DecodeJpeg/contents:0"
tensor_name_input_image = "DecodeJpeg:0"
tensor_name_resized_image = "ResizeBilinear:0"
tensor_name_softmax = "softmax:0"
tensor_name_softmax_logits = "softmax/logits:0"

sess = tf.Session()

tensor_input_jpeg = sess.graph.get_tensor_by_name(name=tensor_name_input_jpeg)
tensor_input_image = sess.graph.get_tensor_by_name(name=tensor_name_input_image)
tensor_resized_image = sess.graph.get_tensor_by_name(name=tensor_name_resized_image)
tensor_softmax = sess.graph.get_tensor_by_name(name=tensor_name_softmax)
tensor_softmax_logits = sess.graph.get_tensor_by_name(name=tensor_name_softmax_logits)

# Adversarial examples

image_data = tf.gfile.FastGFile(os.path.join('inception', 'cropped_panda.jpg'), 'rb').read()
image = sess.run(tensor_resized_image, feed_dict={'DecodeJpeg/contents:0': image_data})

with sess.graph.as_default():
    # target class
    cls_target = tf.placeholder(dtype=tf.int32)

    # loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[cls_target], logits=tensor_softmax_logits)

    # gradients
    gradients = tf.gradients(ys=loss, xs=tensor_resized_image)

# predictions = sess.run(tensor_softmax, feed_dict={'DecodeJpeg/contents:0': image_data})
predictions = sess.run(tensor_softmax, feed_dict={tensor_name_resized_image: image})
predictions = np.squeeze(predictions)

cls_origin = np.argmax(predictions)        # 如果不squeeze, cls = [uid]
namelookup = inception.NameLookup()

noise = 0.0
classes = 100

name_origin = namelookup.cls_to_name(cls_origin).split(',')[0]
name_target = namelookup.cls_to_name(classes).split(',')[0]

for i in range(1000):
    noisy_image = image + noise

    feed_dict = {cls_target: classes, tensor_resized_image: noisy_image}

    pred, grad = sess.run([tensor_softmax, gradients], feed_dict=feed_dict)

    # print(np.array(grad).shape)                 # (1, 1, 299, 299, 3)
    # print(np.squeeze(np.array(grad)).shape)     # (299, 299, 3)

    grad = np.squeeze(np.array(grad))

    grad_absmax = np.abs(grad).max()

    if grad_absmax < 1e-10:
        grad_absmax = 1e-10

    step_size = 7 / grad_absmax

    pred = np.squeeze(pred)
    score = pred[classes]

    if score > 0.99:
        score_origin = pred[cls_origin]
        break
    else:
        noise -= step_size * grad
        noise = np.clip(noise, a_min=-3.0, a_max=3.0)

fig, axes = plt.subplots(1, 2)

ax = axes.flat[0]
msg = "Original Image: \n{0} ({1:.2%})"
xlabel = msg.format(name_origin, predictions[cls_origin])
ax.imshow(np.squeeze(image) / 255.0, interpolation='spline16')
ax.set_xlabel(xlabel)

ax = axes.flat[1]
msg = "Image + Noise:\n{0} ({1:.2%})\n{2} ({3:.2%})"
xlabel = msg.format(name_origin, score_origin, name_target, score)
ax.imshow(np.squeeze(noisy_image) / 255.0, interpolation='spline16')
ax.set_xlabel(xlabel)

plt.show()