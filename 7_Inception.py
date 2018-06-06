import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
# from IPython.display import Image, display
import inception
inception.maybe_download()

model = inception.Inception()

def classify(image_path):

    display(Image(image_path))
    pred = model.classify(image_path=image_path)
    model.print_scores(pred=pred, k=10, only_first_name=True)

image_path = os.path.join(inception.data_dir, 'cropped_panda.jpg')
classify(image_path=image_path)

classify(image_path="images/parrot.jpg")

def plot_resized_image(image_path):
    # Get the resized image from the Inception model
    resized_image = model.get_resized_image(image_path=image_path)

    plt.imshow(resized_image, interpolation='nearest')
    plt.show()

plot_resized_image(image_path='images/parrot.jpg')

# 鹦鹉（裁剪图像，上方）
classify(image_path="images/parrot_cropped1.jpg")

# 鹦鹉（裁剪图像，中间）
classify(image_path="images/parrot_cropped2.jpg")

# 鹦鹉（裁剪图像，底部）
classify(image_path="images/parrot_cropped3.jpg")

# 鹦鹉（填充图像）
classify(image_path="images/parrot_padded.jpg")

# Elon Mush（299x299像素）
classify(image_path="images/elon_musk.jpg")

# Elon Mush（100x100像素）
classify(image_path="images/elon_musk_100x100.jpg")

plot_resized_image(image_path="images/elon_mush_100x100.jpg")

# 查理与巧克力工厂
classify(image_path="images/willy_wonka_old.jpg")
classify(image_path="images/willy_wonka_new.jpg")