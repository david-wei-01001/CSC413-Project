import tensorflow as tf
import numpy as np
import os
import cv2

# Define the paths to the content and style images
style1_path = "style1/"
style2_path = "style2/"

# Define the path to the output folder
out_path = "out_image/"

# Define the size of the generated image
img_size = 256

# Define the weights to use for style mixing
alpha = 0.5
beta = 1 - alpha

# Download and setup the VGG model
vgg_url = "https://tfhub.dev/google/tf2-preview/vgg19/feature_vector/4"
vgg_model = tf.keras.Sequential([
    hub.KerasLayer(vgg_url, input_shape=(img_size, img_size, 3))
])

# Define a function to load and preprocess images
def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = img[np.newaxis, :]
    return img

# Load the content and style images
content_img = load_image("content.jpg")
style1_img = load_image(style1_path + os.listdir(style1_path)[0])
style2_img = load_image(style2_path + os.listdir(style2_path)[0])

# Get the style and content features from the VGG model
style1_features = vgg_model(style1_img)
style2_features = vgg_model(style2_img)
content_features = vgg_model(content_img)

# Mix the style features
mixed_features = alpha * style1_features + beta * style2_features

# Generate the output image from the mixed features and content features
generated_img = vgg_model(mixed_features, training=False)
generated_img = tf.squeeze(generated_img)
generated_img = tf.clip_by_value(generated_img, 0, 1)

# Save the generated image to disk
cv2.imwrite(out_path + "generated.jpg", generated_img.numpy() * 255)
