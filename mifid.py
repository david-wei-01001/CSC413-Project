import os
import glob
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from scipy import linalg

# Define InceptionV3 model as a global variable
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

def load_images(image_paths):
    """Loads images as an array of numpy arrays"""
    images = []
    for path in image_paths:
        # Load image as PIL Image object
        img = load_img(path, target_size=(256, 256))
        # Convert PIL Image object to numpy array
        img = img_to_array(img)
        images.append(img)
    return np.asarray(images)

def calculate_activation_statistics(images, model):
    """Calculates the activation statistics for the InceptionV3 network for a given set of images."""
    # Reshape images to 299x299, as required by InceptionV3
    images = tf.image.resize(images, (299, 299), method=tf.image.ResizeMethod.BILINEAR)
    # Preprocess images using the preprocess_input function of InceptionV3
    images = preprocess_input(images)
    # Pass images through the InceptionV3 network to get activations
    activations = model.predict(images)
    # Flatten activations
    activations = activations.reshape(activations.shape[0], -1)
    # Calculate mean and covariance of activations
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    """Calculates the Fréchet distance between two multivariate Gaussians with the given means and covariances."""
    # Calculate the square root of the product of the two covariances using the linalg.sqrtm function
    cov_sqrt = linalg.sqrtm(np.dot(sigma1, sigma2))
    # Calculate the squared distance between the means using the np.sum function
    mu_diff = mu1 - mu2
    mu_diff_squared = np.dot(mu_diff, mu_diff)
    # Calculate the Fréchet distance using the np.real function to take the real part of the result
    fid = np.real(mu_diff_squared + np.trace(sigma1 + sigma2 - 2 * cov_sqrt))
    return fid


def calculate_mifid(images1_path, images2_path):
    """Calculates the MiFID score between two sets of images for a given InceptionV3 model."""
    # Load images from the provided image paths
    images1 = load_images(glob.glob(os.path.join(images1_path, '*.jpg')))
    images2 = load_images(glob.glob(os.path.join(images2_path, '*.jpg')))
    # Calculate activation statistics for both sets of images using the calculate_activation_statistics function
    mu1, sigma1 = calculate_activation_statistics(images1, model)
    mu2, sigma2 = calculate_activation_statistics(images2, model)
    # Calculate the Fréchet distance between the two multivariate Gaussians using the calculate_frechet_distance function
    mifid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return mifid


# Calculate MiFID score for GAN output
gan_mifid = calculate_mifid('test_monet', 'GAN_out')
print(f"MiFID score for GAN output is: {gan_mifid}")

# Calculate MiFID score for NST output
nst_mifid = calculate_mifid('test_monet', 'NST_out')
print(f"MiFID score for GAN output is: {nst_mifid}")
