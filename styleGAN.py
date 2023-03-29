"""
The styleGAN pretrained Model
"""
import tensorflow_hub as hub
import numpy as np
import os
import cv2
from typing import Any

# Define the paths to the style images
style1_path = "style1/"
style2_path = "style2/"

# Define the path to the output folder
out_path = "out_image/"


# Define the size of the generated image
img_size = 256

# Define the weights to use for style mixing
alpha = 0.5
beta = 1 - alpha

# Model URLs
styleGANs = {
    1: "https://tfhub.dev/google/stylegan/1",
    2: "https://tfhub.dev/google/stylegan2/1",
    3: "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1"
}


# Define a function to load and preprocess images
def load_image(img_path: str) -> Any:
    """
    Load and preprocess the image.

    :param img_path: the path of image to load
    :return: the processed image
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img).astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def process_data() -> Any:
    """
    Load the 2 style images from their corresponding path
    :return: style1_img is the image of the first style, style2_img is the image of the second style
    """
    style1_img = load_image(style1_path)
    style2_img = load_image(style2_path)
    return style1_img, style2_img


class styleGAN:
    """
    The styleGAN pretrained-model.

    Instance Variables:
        - url: the url of the model
        - model: the model of the pretrained model
        - mixed_features: the mixed feature after mixing style 1 and 2
    """
    _url: str
    _model: Any
    _mixed_features: Any

    def __init__(self, url: str) -> None:
        """
        Initialization

        :param url: the url of the pretrained_model
        """
        self._url = url

    def _load_model(self) -> None:
        """
        load the model
        """
        self._model = hub.load(self._url)

    def _mix(self, style1, style2) -> None:
        """
        Mix the style of style1 and style2 corresponding to hyperparameter alpha and beta
        and store the style into self.mixed_features

        :param style1: source of style1
        :param style2: source of style2
        """
        style1_features = self._model(style1)['4']
        style2_features = self._model(style2)['4']
        self._mixed_features = alpha * style1_features + beta * style2_features

    def _generate_and_save(self) -> None:
        """
        Use self.model and self.mixed_features to generate the mixed pictures and store them
        """
        generated_images = self._model(self._mixed_features)['default']
        for i in range(generated_images.shape[0]):
            img = generated_images[i].numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = (img * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(out_path, f"generated_image_{i}.jpg"), img)

    def generate(self, style1, style2) -> None:
        """
        Now perform image generation using the model!

        :param style1: source of style1
        :param style2: source of style2
        """
        self._load_model()
        self._mix(style1, style2)
        self._generate_and_save()


if __name__ == "__main__":
    model = styleGAN(styleGANs[1])
    style1_out, style2_out = process_data()
    model.generate(style1_out, style2_out)
