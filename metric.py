"""
Several Metrics to measure how good the style mixing is.
"""
from typing import Any
import torch
import torchvision
import pytorch_fid
import numpy as np
from skimage.measure import compare_ssim


# The paths that stores the style images
style1_path = "style1/"
style2_path = "style2/"

# The paths that stores the generated images
out_path = "out_image/"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_images(generate_path: str, style1: str, style2: str):
    """
    1. Create a dataset that contains a representative set of images from "style1" and "style2"
    folders, and load them into a dataloader.
    2.Create a second dataset that contains the generated style-mixed images and load them into
    a dataloader.

    :param generate_path: The paths that stores the generated images
    :param style1: The paths that stores images of one style
    :param style2: The paths that stores images of the other style
    :return: the dataloaders
    """
    # Define the transform for image resizing and normalization
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load style1 images into dataset1
    dataset1 = torchvision.datasets.ImageFolder(root=style1, transform=transform)

    # Load style2 images into dataset2
    dataset2 = torchvision.datasets.ImageFolder(root=style2, transform=transform)

    # Load generated images into dataset3
    dataset3 = torchvision.datasets.ImageFolder(root=generate_path, transform=transform)

    # Create dataloaders for each dataset
    dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=64, shuffle=False, num_workers=4)
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=64, shuffle=False, num_workers=4)
    dataloader3 = torch.utils.data.DataLoader(dataset3, batch_size=64, shuffle=False, num_workers=4)
    return dataloader1, dataloader2, dataloader3


class metric:
    """A metric that quantify the quality of the generated style mixing images."""
    device: Any

    def __init__(self) -> None:
        """
        Initialization.
        """
        self.device = DEVICE

    def calculate(self, generate: Any, style1: Any, style2: Any) -> float:
        """
        Calculate the metric score.

        :param generate: dataloaders for generated images
        :param style1: dataloaders for images of one style
        :param style2: dataloaders for images of the other style
        :return: the metriic score
        """
        raise NotImplementedError

    #     TODO: Add any mutually used functions here


class FID(metric):
    """This calculates the FID (Frechet Inception Distance) score
     of the generated style mixing images."""

    def calculate(self, generate: Any, style1: Any, style2: Any) -> float:
        """
        Calculate the FID score.

        :param generate: dataloaders for generated images
        :param style1: dataloaders for images of one style
        :param style2: dataloaders for images of the other style
        :return: the FID score
        """
        # Load the Inception v3 model
        model = torchvision.models.inception_v3(pretrained=True)
        model.to(self.device)
        model.eval()

        # Calculate the mean and covariance of the Inception v3 feature embeddings for style1 images
        style1_features = []
        for images, _ in style1:
            images = images.to(self.device)
            with torch.no_grad():
                features = model(images)[0]
            style1_features.append(features.cpu().numpy())
        style1_features = torch.from_numpy(np.concatenate(style1_features, axis=0))
        style1_mean, style1_cov = pytorch_fid.calculate_activation_statistics(style1_features)

        # Calculate the mean and covariance of the Inception v3 feature embeddings for style2 images
        style2_features = []
        for images, _ in style2:
            images = images.to(self.device)
            with torch.no_grad():
                features = model(images)[0]
            style2_features.append(features.cpu().numpy())
        style2_features = torch.from_numpy(np.concatenate(style2_features, axis=0))
        style2_mean, style2_cov = pytorch_fid.calculate_activation_statistics(style2_features)

        # Calculate the mean and covariance of
        # the Inception v3 feature embeddings for generated images
        generate_features = []
        for images, _ in generate:
            images = images.to(self.device)
            with torch.no_grad():
                features = model(images)[0]
            generate_features.append(features.cpu().numpy())
        generate_features = torch.from_numpy(np.concatenate(generate_features, axis=0))
        generate_mean, generate_cov = pytorch_fid.calculate_activation_statistics(generate_features)

        # Calculate the FID score
        fid_score = pytorch_fid.calculate_frechet_distance(
            style1_mean, style1_cov, generate_mean, generate_cov
        ) + pytorch_fid.calculate_frechet_distance(
            style2_mean, style2_cov, generate_mean, generate_cov
        )

        return fid_score.item()


class SSIM(metric):
    """This calculates the SSIM (Structural Similarity Index) score
     of the generated style mixing images."""

    def calculate(self, generate: Any, style1: Any, style2: Any) -> float:
        """
        Calculate the SSIM score.

        :param generate: dataloaders for generated images
        :param style1: dataloaders for images of one style
        :param style2: dataloaders for images of the other style
        :return: the SSIM score
        """
        ssim_total = 0.0
        num_images = 0
        for generated_images, style1_images, style2_images in zip(generate, style1, style2):
            generated_images = generated_images.to(self.device)
            style1_images = style1_images.to(self.device)
            style2_images = style2_images.to(self.device)
            generated_images = (generated_images + 1) / 2  # convert range from [-1, 1] to [0, 1]
            style1_images = (style1_images + 1) / 2
            style2_images = (style2_images + 1) / 2
            for i in range(generated_images.size(0)):
                num_images += 1
                generated_image = generated_images[i].detach().cpu().numpy().transpose((1, 2, 0))
                style1_image = style1_images[i].detach().cpu().numpy().transpose((1, 2, 0))
                style2_image = style2_images[i].detach().cpu().numpy().transpose((1, 2, 0))
                generated_image = (generated_image * 255).astype(np.uint8)
                style1_image = (style1_image * 255).astype(np.uint8)
                style2_image = (style2_image * 255).astype(np.uint8)
                ssim1 = compare_ssim(generated_image, style1_image, multichannel=True)
                ssim2 = compare_ssim(generated_image, style2_image, multichannel=True)
                ssim_total += max(ssim1, ssim2)
        ssim_score = ssim_total / num_images
        return ssim_score


class MSE(metric):
    """This calculates the MSE (Mean Squared Error) score of the generated style mixing images."""

    def calculate(self, generate: Any, style1: Any, style2: Any) -> float:
        """
        Calculate the MSE score.

        :param generate: dataloaders for generated images
        :param style1: dataloaders for images of one style
        :param style2: dataloaders for images of the other style
        :return: the MSE score
        """
        mse_score = 0.0
        num_images = 0

        for gen_images, style1_images, style2_images in zip(generate, style1, style2):
            gen_images = gen_images.to(self.device)
            style1_images = style1_images.to(self.device)
            style2_images = style2_images.to(self.device)

            # Compute the MSE for style 1 images
            mse_score += torch.mean((gen_images - style1_images) ** 2)
            num_images += gen_images.shape[0]

            # Compute the MSE for style 2 images
            mse_score += torch.mean((gen_images - style2_images) ** 2)
            num_images += gen_images.shape[0]

        return mse_score.item() / num_images
